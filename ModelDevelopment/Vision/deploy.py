"""
deploy.py - Deploy Vision model to Vertex AI Model Registry
"""
import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

from google.cloud import aiplatform
from google.cloud import storage
import tensorflow as tf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def deploy_vision_model(
    model_path: str,
    metadata_path: str,
    dataset: str,
    model_name: str,
    project_id: str = None,
    region: str = "us-central1",
    bucket_name: str = None
):
    """
    Deploy Vision model to Vertex AI Model Registry.
    
    Args:
        model_path: Path to .keras model file
        metadata_path: Path to training_metadata.json
        dataset: Dataset name (tb or lung_cancer_ct_scan)
        model_name: Model name (e.g., CNN_ResNet18)
        project_id: GCP project ID
        region: GCP region
        bucket_name: GCS bucket name
    """
    # Get environment variables if not provided
    if project_id is None:
        project_id = os.getenv("GCP_PROJECT_ID")
        if not project_id:
            raise ValueError("GCP_PROJECT_ID not set")
    
    if bucket_name is None:
        bucket_name = os.getenv("GCS_BUCKET_NAME")
        if not bucket_name:
            raise ValueError("GCS_BUCKET_NAME not set")
    
    # Initialize clients
    aiplatform.init(project=project_id, location=region)
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    metrics = metadata.get('metrics', {})
    test_accuracy = metrics.get('test_accuracy', 0.0)
    
    logger.info(f"Deploying model: {model_name}")
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Test Accuracy: {test_accuracy}")
    
    # Convert .keras to SavedModel format
    logger.info("Loading Keras model...")
    model = tf.keras.models.load_model(model_path)
    
    # Create temporary directory for SavedModel
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        saved_model_path = Path(temp_dir) / "saved_model"
        logger.info(f"Converting to SavedModel format...")
        # Keras 3: Use tf.saved_model.save() directly instead of model.save() with save_format
        tf.saved_model.save(model, str(saved_model_path))
        
        # Upload SavedModel to GCS
        build_id = os.getenv("BUILD_ID", datetime.now().strftime("%Y%m%d%H%M%S"))
        gcs_model_dir = f"vision/models/{build_id}/saved_model"
        
        logger.info(f"Uploading SavedModel to gs://{bucket_name}/{gcs_model_dir}/")
        for root, dirs, files in os.walk(saved_model_path):
            for file in files:
                local_file = Path(root) / file
                relative_path = local_file.relative_to(saved_model_path)
                gcs_path = f"{gcs_model_dir}/{relative_path}"
                
                blob = bucket.blob(gcs_path)
                blob.upload_from_filename(str(local_file))
        
        artifact_uri = f"gs://{bucket_name}/{gcs_model_dir}/"
        logger.info(f"Model uploaded to: {artifact_uri}")
    
    # Register model in Vertex AI
    display_name = f"medscan-vision-{dataset}-{model_name.lower()}"
    description = (
        f"Vision model for {dataset} detection. "
        f"Model: {model_name}. "
        f"Test Accuracy: {test_accuracy:.4f}"
    )
    
    labels = {
        "model-type": "vision",
        "dataset": dataset,
        "model-name": model_name.lower(),
        "framework": "tensorflow"
    }
    
    logger.info(f"Registering model in Vertex AI: {display_name}")
    model = aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=artifact_uri,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-13:latest",
        description=description,
        labels=labels
    )
    
    logger.info(f"Model registered: {model.resource_name}")
    logger.info(f"Model ID: {model.resource_name.split('/')[-1]}")
    
    # Save deployment info
    deployment_info = {
        "model_resource_name": model.resource_name,
        "artifact_uri": artifact_uri,
        "metadata": metadata,
        "deployment_timestamp": datetime.utcnow().isoformat() + "Z"
    }
    
    deployment_info_path = f"vision/deployments/{build_id}/deployment_info.json"
    blob = bucket.blob(deployment_info_path)
    blob.upload_from_string(json.dumps(deployment_info, indent=2))
    logger.info(f"Deployment info saved to: gs://{bucket_name}/{deployment_info_path}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Deploy Vision model to Vertex AI")
    parser.add_argument("--model_path", required=True, help="Path to .keras model file")
    parser.add_argument("--metadata_file", required=True, help="Path to training_metadata.json")
    parser.add_argument("--dataset", required=True, help="Dataset name (tb or lung_cancer_ct_scan)")
    parser.add_argument("--model_name", required=True, help="Model name (e.g., CNN_ResNet18)")
    parser.add_argument("--project_id", help="GCP project ID (or set GCP_PROJECT_ID env var)")
    parser.add_argument("--region", default="us-central1", help="GCP region")
    parser.add_argument("--bucket_name", help="GCS bucket name (or set GCS_BUCKET_NAME env var)")
    
    args = parser.parse_args()
    
    try:
        deploy_vision_model(
            model_path=args.model_path,
            metadata_path=args.metadata_file,
            dataset=args.dataset,
            model_name=args.model_name,
            project_id=args.project_id,
            region=args.region,
            bucket_name=args.bucket_name
        )
        logger.info("Deployment completed successfully")
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
