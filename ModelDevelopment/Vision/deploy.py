"""
deploy.py - Deploy Vision models to Vertex AI
"""
import os
import logging
from pathlib import Path
from typing import Optional
from google.cloud import aiplatform
from google.cloud import storage
import json
import tempfile
import shutil
from tensorflow import keras

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VisionModelDeployer:
    """Deploy Vision models to Vertex AI"""
    
    def __init__(
        self,
        project_id: str = None,
        region: str = "us-central1",
        bucket_name: str = None
    ):
        if project_id is None:
            project_id = os.getenv("GCP_PROJECT_ID")
            if not project_id:
                raise ValueError(
                    "GCP_PROJECT_ID not set. Please set it as an environment variable "
                    "or pass it to VisionModelDeployer(project_id='...')"
                )
        if bucket_name is None:
            bucket_name = os.getenv("GCS_BUCKET_NAME")
            if not bucket_name:
                raise ValueError(
                    "GCS_BUCKET_NAME not set. Please set it as an environment variable "
                    "or pass it to VisionModelDeployer(bucket_name='...')"
                )
        self.project_id = project_id
        self.region = region
        self.bucket_name = bucket_name
        
        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=region)
        
        # Initialize GCS client
        self.storage_client = storage.Client(project=project_id)
        self.bucket = self.storage_client.bucket(bucket_name)
    
    def _upload_directory(self, local_dir: Path, gcs_prefix: str) -> str:
        """
        Upload a local directory (recursively) to GCS under gcs_prefix.
        
        Args:
            local_dir: Path to local directory
            gcs_prefix: GCS prefix (e.g., 'vision/models/tb/CNN_ResNet50/saved_model')
        
        Returns:
            GCS URI of the uploaded directory
        """
        local_dir = Path(local_dir)
        if not local_dir.exists() or not local_dir.is_dir():
            raise ValueError(f"Local directory does not exist: {local_dir}")
        
        for path in local_dir.rglob("*"):
            if path.is_file():
                rel_path = path.relative_to(local_dir)
                blob_path = f"{gcs_prefix}/{rel_path.as_posix()}"
                blob = self.bucket.blob(blob_path)
                blob.upload_from_filename(str(path))
        return f"gs://{self.bucket_name}/{gcs_prefix}"
    
    def upload_model_to_gcs(
        self,
        local_model_path: str,
        model_name: str,
        dataset_name: str
    ) -> str:
        """
        Upload trained model to GCS.
        
        Args:
            local_model_path: Path to local .keras model file
            model_name: Name of model (e.g., 'CNN_ResNet50')
            dataset_name: Dataset name (e.g., 'tb')
            
        Returns:
            GCS URI of uploaded model
        """
        # Create GCS path: vision/models/{dataset}/{model_name}/model.keras
        gcs_path = f"vision/models/{dataset_name}/{model_name}/model.keras"
        
        logger.info(f"Uploading model to gs://{self.bucket_name}/{gcs_path}")
        
        blob = self.bucket.blob(gcs_path)
        blob.upload_from_filename(local_model_path)
        
        gcs_uri = f"gs://{self.bucket_name}/{gcs_path}"
        logger.info(f"Model uploaded to {gcs_uri}")
        
        return gcs_uri
    
    def register_model(
        self,
        display_name: str,
        model_gcs_uri: str,
        description: str,
        labels: Optional[dict] = None
    ) -> aiplatform.Model:
        """
        Register model in Vertex AI Model Registry.
        
        Args:
            display_name: Display name for model
            model_gcs_uri: GCS URI of model artifact
            description: Model description
            labels: Optional labels dict
            
        Returns:
            Vertex AI Model object
        """
        logger.info(f"Registering model: {display_name}")
        
        # Create model
        model = aiplatform.Model.upload(
            display_name=display_name,
            artifact_uri=model_gcs_uri,
            serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-13:latest",
            description=description,
            labels=labels or {}
        )
        
        logger.info(f"Model registered: {model.resource_name}")
        return model
    
    def deploy_model(
        self,
        model: aiplatform.Model,
        endpoint_display_name: str,
        machine_type: str = "n1-standard-4",
        min_replica_count: int = 1,
        max_replica_count: int = 3
    ) -> aiplatform.Endpoint:
        """
        Deploy model to Vertex AI endpoint.
        
        Args:
            model: Vertex AI Model object
            endpoint_display_name: Name for endpoint
            machine_type: Machine type for deployment
            min_replica_count: Minimum replicas
            max_replica_count: Maximum replicas
            
        Returns:
            Vertex AI Endpoint object
        """
        logger.info(f"Deploying model to endpoint: {endpoint_display_name}")
        
        # Check if endpoint exists
        endpoints = aiplatform.Endpoint.list(
            filter=f'display_name="{endpoint_display_name}"'
        )
        
        if endpoints:
            endpoint = endpoints[0]
            logger.info(f"Using existing endpoint: {endpoint.resource_name}")
        else:
            endpoint = aiplatform.Endpoint.create(
                display_name=endpoint_display_name
            )
            logger.info(f"Created new endpoint: {endpoint.resource_name}")
        
        # Deploy model
        model.deploy(
            endpoint=endpoint,
            deployed_model_display_name=model.display_name,
            machine_type=machine_type,
            min_replica_count=min_replica_count,
            max_replica_count=max_replica_count,
            traffic_percentage=100,
            sync=True
        )
        
        logger.info(f"Model deployed successfully")
        return endpoint
    
    def deploy_vision_model(
        self,
        model_path: str,
        dataset_name: str,
        model_name: str,
        test_accuracy: float,
        metadata: dict
    ):
        """
        Complete deployment workflow for vision model.
        
        Args:
            model_path: Path to trained model
            dataset_name: Dataset name (tb/lung_cancer)
            model_name: Model architecture name
            test_accuracy: Model test accuracy
            metadata: Model metadata dict
        """
        logger.info("="*80)
        logger.info("Starting Vision Model Deployment")
        logger.info("="*80)
        
        # 1. Upload Keras model file to GCS for archival
        gcs_keras_uri = self.upload_model_to_gcs(model_path, model_name, dataset_name)
        
        # 2. Export TensorFlow SavedModel and upload directory to GCS for Vertex AI
        tmp_dir = Path(tempfile.mkdtemp(prefix="export_saved_model_"))
        try:
            logger.info("Exporting Keras model to TensorFlow SavedModel format...")
            model = keras.models.load_model(model_path)
            export_dir = tmp_dir / "saved_model"
            model.export(str(export_dir))
            
            # Upload exported directory to GCS under saved_model/
            saved_model_prefix = f"vision/models/{dataset_name}/{model_name}/saved_model"
            logger.info(f"Uploading SavedModel directory to gs://{self.bucket_name}/{saved_model_prefix}")
            gcs_saved_model_uri = self._upload_directory(export_dir, saved_model_prefix)
        finally:
            # Clean up temp directory
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass
        
        # 3. Register in Model Registry (use SavedModel directory as artifact_uri)
        display_name = f"{dataset_name}_{model_name}".replace("_", "-")
        description = (
            f"Vision model for {dataset_name} classification. "
            f"Architecture: {model_name}. "
            f"Test Accuracy: {test_accuracy:.4f}"
        )
        
        labels = {
            "dataset": dataset_name.replace("_", "-"),
            "architecture": model_name.lower().replace("_", "-"),
            "framework": "tensorflow",
            "accuracy": str(int(test_accuracy * 100))
        }
        
        model = self.register_model(
            display_name=display_name,
            model_gcs_uri=gcs_saved_model_uri,
            description=description,
            labels=labels
        )
        
        # 4. Deploy to endpoint
        endpoint_name = f"{dataset_name}-vision-endpoint".replace("_", "-")
        endpoint = self.deploy_model(
            model=model,
            endpoint_display_name=endpoint_name
        )
        
        # 5. Save deployment info
        deployment_info = {
            "model_resource_name": model.resource_name,
            "endpoint_resource_name": endpoint.resource_name,
            "gcs_saved_model_uri": gcs_saved_model_uri,
            "gcs_keras_uri": gcs_keras_uri,
            "test_accuracy": test_accuracy,
            "metadata": metadata
        }
        
        # Upload deployment info to GCS
        info_path = f"vision/deployments/{dataset_name}/{model_name}/deployment_info.json"
        blob = self.bucket.blob(info_path)
        blob.upload_from_string(json.dumps(deployment_info, indent=2))
        
        logger.info("="*80)
        logger.info("Deployment Complete!")
        logger.info(f"Model: {model.resource_name}")
        logger.info(f"Endpoint: {endpoint.resource_name}")
        logger.info("="*80)
        
        return model, endpoint


def main():
    """Main deployment function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Deploy Vision model to Vertex AI')
    parser.add_argument('--model_path', required=True, help='Path to .keras model')
    parser.add_argument('--dataset', required=True, choices=['tb', 'lung_cancer_ct_scan'])
    parser.add_argument('--model_name', required=True, help='Model architecture name')
    parser.add_argument('--test_accuracy', type=float, required=True)
    parser.add_argument('--metadata_file', help='Path to metadata JSON file')
    
    args = parser.parse_args()
    
    # Load metadata
    metadata = {}
    if args.metadata_file:
        with open(args.metadata_file, 'r') as f:
            metadata = json.load(f)
    
    # Deploy
    deployer = VisionModelDeployer()
    deployer.deploy_vision_model(
        model_path=args.model_path,
        dataset_name=args.dataset,
        model_name=args.model_name,
        test_accuracy=args.test_accuracy,
        metadata=metadata
    )


if __name__ == "__main__":
    main()