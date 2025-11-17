"""
deploy.py - Deploy RAG model to Vertex AI with vLLM
Optimized for cost-effective deployment
"""
import os
import logging
import json
from pathlib import Path
from google.cloud import aiplatform
from google.cloud import storage
from typing import Dict, Any, Optional
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGModelDeployer:
    """Deploy RAG model to Vertex AI with vLLM"""
    
    VLLM_DOCKER_URI = "us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve"
    
    # Cost-effective machine types
    MACHINE_TYPES = {
        "small": {
            "type": "n1-standard-4",
            "accelerator": "NVIDIA_TESLA_T4",
            "accelerator_count": 1,
            "cost_per_hour": 0.35
        },
        "medium": {
            "type": "n1-standard-8", 
            "accelerator": "NVIDIA_TESLA_T4",
            "accelerator_count": 1,
            "cost_per_hour": 0.50
        },
        "large": {
            "type": "a2-highgpu-1g",
            "accelerator": "NVIDIA_TESLA_A100",
            "accelerator_count": 1,
            "cost_per_hour": 3.67
        }
    }
    
    def __init__(
        self,
        project_id: str = None,
        region: str = "us-central1",
        bucket_name: str = None,
        staging_bucket: str = None
    ):
        if project_id is None:
            project_id = os.getenv("GCP_PROJECT_ID")
            if not project_id:
                raise ValueError("GCP_PROJECT_ID not set")
        if bucket_name is None:
            bucket_name = os.getenv("GCS_BUCKET_NAME")
            if not bucket_name:
                raise ValueError("GCS_BUCKET_NAME not set")
        
        self.project_id = project_id
        self.region = region
        self.bucket_name = bucket_name
        self.staging_bucket = staging_bucket or f"gs://{bucket_name}/staging"
        
        aiplatform.init(
            project=project_id,
            location=region,
            staging_bucket=self.staging_bucket
        )
        
        self.storage_client = storage.Client(project=project_id)
        self.bucket = self.storage_client.bucket(bucket_name)
        
        logger.info(f"Initialized Vertex AI (project: {project_id}, region: {region})")
    
    def _resolve_path(self, path: str) -> str:
        """Resolve path to GCS URI or upload if local"""
        if path.startswith('gs://'):
            return path
        
        local_path = Path(path)
        if not local_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        filename = local_path.name
        gcs_path = f"RAG/models/uploads/{filename}"
        
        logger.info(f"Uploading {filename} to GCS")
        blob = self.bucket.blob(gcs_path)
        blob.upload_from_filename(str(local_path))
        
        uri = f"gs://{self.bucket_name}/{gcs_path}"
        logger.info(f"Uploaded: {uri}")
        return uri
    
    def upload_model_artifacts(
        self,
        config_path: str,
        index_path: str,
        embeddings_path: str
    ) -> Dict[str, str]:
        """Upload RAG artifacts to GCS under latest/"""
        logger.info("Processing RAG artifacts")
        
        uris = {}
        
        config_uri = self._resolve_path(config_path)
        index_uri = self._resolve_path(index_path)
        embeddings_uri = self._resolve_path(embeddings_path)
        
        for name, source_uri in [
            ("config", config_uri),
            ("index", index_uri),
            ("embeddings", embeddings_uri)
        ]:
            source_parts = source_uri.replace('gs://', '').split('/', 1)
            source_bucket_name = source_parts[0]
            source_blob_path = source_parts[1]
            
            dest_blob_path = f"RAG/models/latest/{name}.{source_blob_path.split('.')[-1]}"
            dest_blob = self.bucket.blob(dest_blob_path)
            
            if source_bucket_name == self.bucket_name:
                source_blob = self.bucket.blob(source_blob_path)
                self.bucket.copy_blob(source_blob, self.bucket, dest_blob.name)
            else:
                source_client = storage.Client()
                source_bucket_obj = source_client.bucket(source_bucket_name)
                source_blob = source_bucket_obj.blob(source_blob_path)
                content = source_blob.download_as_bytes()
                dest_blob.upload_from_string(content)
            
            uris[name] = f"gs://{self.bucket_name}/{dest_blob_path}"
            logger.info(f"Copied {name}: {uris[name]}")
        
        return uris
    
    def _get_vllm_args(self, config: Dict[str, Any]) -> list:
        """Generate vLLM server arguments from config"""
        model_name = config.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct")
        
        vllm_args = [
            "--host=0.0.0.0",
            "--port=7080",
            f"--model={model_name}",
            "--tensor-parallel-size=1",
            "--swap-space=16",
            "--gpu-memory-utilization=0.85",
            "--disable-log-stats",
            "--max-model-len=8192",
            "--trust-remote-code"
        ]
        
        return vllm_args
    
    def _select_machine_type(self, model_size: str) -> Dict[str, Any]:
        """Select cost-effective machine type based on model size"""
        model_size_lower = model_size.lower()
        
        if "14b" in model_size_lower or "13b" in model_size_lower:
            machine = "medium"
        elif "7b" in model_size_lower or "8b" in model_size_lower:
            machine = "small"
        else:
            machine = "small"
        
        config = self.MACHINE_TYPES[machine]
        logger.info(f"Selected {machine} machine: {config['type']} with {config['accelerator']}")
        logger.info(f"Estimated cost: ${config['cost_per_hour']:.2f}/hour")
        
        return config
    
    def deploy_rag_endpoint(
        self,
        config_path: str,
        index_path: str,
        embeddings_path: str,
        metadata_path: str,
        endpoint_display_name: str = "medscan-rag-endpoint",
        machine_size: str = "small"
    ) -> aiplatform.Endpoint:
        """
        Deploy RAG model to Vertex AI endpoint with vLLM.
        
        Args:
            config_path: Path to RAG config JSON
            index_path: Path to FAISS index
            embeddings_path: Path to embeddings JSON
            metadata_path: Path to metadata JSON
            endpoint_display_name: Name for endpoint
            machine_size: 'small', 'medium', or 'large'
        
        Returns:
            Deployed Vertex AI Endpoint
        """
        logger.info("=" * 80)
        logger.info("Starting RAG Vertex AI Deployment")
        logger.info("=" * 80)
        
        if metadata_path.startswith('gs://'):
            parts = metadata_path.replace('gs://', '').split('/', 1)
            bucket = self.storage_client.bucket(parts[0])
            blob = bucket.blob(parts[1])
            metadata = json.loads(blob.download_as_text())
        else:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        config_uri = self._resolve_path(config_path)
        config_blob = self.bucket.blob(config_uri.replace(f'gs://{self.bucket_name}/', ''))
        config = json.loads(config_blob.download_as_text())
        
        uris = self.upload_model_artifacts(config_path, index_path, embeddings_path)
        
        model_name = config.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct")
        vllm_args = self._get_vllm_args(config)
        
        machine_config = self._select_machine_type(model_name)
        
        logger.info(f"Deploying model: {model_name}")
        logger.info(f"vLLM args: {' '.join(vllm_args)}")
        
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
            logger.info(f"Created endpoint: {endpoint.resource_name}")
        
        model_display_name = f"medscan-rag-{model_name.split('/')[-1].lower().replace('.', '-')}"
        
        model = aiplatform.Model.upload(
            display_name=model_display_name,
            serving_container_image_uri=self.VLLM_DOCKER_URI,
            serving_container_command=["python", "-m", "vllm.entrypoints.api_server"],
            serving_container_args=vllm_args,
            serving_container_ports=[7080],
            serving_container_predict_route="/generate",
            serving_container_health_route="/ping",
            serving_container_environment_variables={
                "HUGGING_FACE_HUB_TOKEN": os.getenv("HF_TOKEN", ""),
                "INDEX_URI": uris['index'],
                "EMBEDDINGS_URI": uris['embeddings'],
                "CONFIG_URI": uris['config']
            },
            description=f"RAG model: {model_name}. Composite score: {metadata.get('performance_metrics', {}).get('composite_score', 0):.4f}"
        )
        
        logger.info(f"Model uploaded: {model.resource_name}")
        
        logger.info("Deploying to endpoint")
        model.deploy(
            endpoint=endpoint,
            deployed_model_display_name=model_display_name,
            machine_type=machine_config['type'],
            accelerator_type=machine_config['accelerator'],
            accelerator_count=machine_config['accelerator_count'],
            min_replica_count=1,
            max_replica_count=1,
            traffic_percentage=100,
            deploy_request_timeout=1800,
            sync=True
        )
        
        deployment_info = {
            "endpoint_resource_name": endpoint.resource_name,
            "endpoint_url": f"https://{self.region}-aiplatform.googleapis.com/v1/{endpoint.resource_name}:predict",
            "model_resource_name": model.resource_name,
            "model_name": model_name,
            "artifact_uris": uris,
            "machine_config": machine_config,
            "vllm_args": vllm_args,
            "metadata": metadata,
            "deployed_at": aiplatform.utils.get_timestamp()
        }
        
        info_blob = self.bucket.blob("RAG/deployments/latest/deployment_info.json")
        info_blob.upload_from_string(json.dumps(deployment_info, indent=2))
        
        logger.info("=" * 80)
        logger.info("RAG Deployment Complete")
        logger.info(f"Endpoint: {endpoint.resource_name}")
        logger.info(f"Model: {model_name}")
        logger.info(f"Machine: {machine_config['type']} + {machine_config['accelerator']}")
        logger.info(f"Estimated cost: ${machine_config['cost_per_hour']:.2f}/hour")
        logger.info(f"Prediction URL: {deployment_info['endpoint_url']}")
        logger.info("=" * 80)
        
        return endpoint


def main():
    """Main deployment function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Deploy RAG to Vertex AI')
    parser.add_argument('--config', required=True, help='Config JSON path')
    parser.add_argument('--index', required=True, help='FAISS index path')
    parser.add_argument('--embeddings', required=True, help='Embeddings JSON path')
    parser.add_argument('--metadata', required=True, help='Metadata JSON path')
    parser.add_argument('--machine-size', default='small', choices=['small', 'medium', 'large'])
    
    args = parser.parse_args()
    
    deployer = RAGModelDeployer()
    endpoint = deployer.deploy_rag_endpoint(
        config_path=args.config,
        index_path=args.index,
        embeddings_path=args.embeddings,
        metadata_path=args.metadata,
        machine_size=args.machine_size
    )
    
    logger.info(f"Deployment successful. Endpoint: {endpoint.resource_name}")


if __name__ == "__main__":
    main()