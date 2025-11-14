"""
deploy.py - Deploy RAG model to Vertex AI
"""
import logging
import json
from pathlib import Path
from google.cloud import aiplatform
from google.cloud import storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGModelDeployer:
    """Deploy RAG model and index to Vertex AI"""
    
    def __init__(
        self,
        project_id: str = "medscanai-476203",
        region: str = "us-central1",
        bucket_name: str = "medscan-data"
    ):
        self.project_id = project_id
        self.region = region
        self.bucket_name = bucket_name
        
        aiplatform.init(project=project_id, location=region)
        self.storage_client = storage.Client(project=project_id)
        self.bucket = self.storage_client.bucket(bucket_name)
    
    def upload_model_artifacts(
        self,
        config_path: str,
        index_path: str,
        embeddings_path: str
    ) -> dict:
        """
        Upload RAG model artifacts to GCS.
        
        Returns:
            Dict with GCS URIs
        """
        logger.info("Uploading RAG model artifacts to GCS")
        
        uris = {}
        
        # Upload config
        config_blob = self.bucket.blob("RAG/models/latest/config.json")
        config_blob.upload_from_filename(config_path)
        uris['config'] = f"gs://{self.bucket_name}/{config_blob.name}"
        
        # Upload FAISS index
        index_blob = self.bucket.blob("RAG/models/latest/index.bin")
        index_blob.upload_from_filename(index_path)
        uris['index'] = f"gs://{self.bucket_name}/{index_blob.name}"
        
        # Upload embeddings
        embeddings_blob = self.bucket.blob("RAG/models/latest/embeddings.json")
        embeddings_blob.upload_from_filename(embeddings_path)
        uris['embeddings'] = f"gs://{self.bucket_name}/{embeddings_blob.name}"
        
        logger.info("Artifacts uploaded successfully")
        return uris
    
    def register_rag_model(
        self,
        display_name: str,
        artifact_uri: str,
        metadata: dict
    ) -> aiplatform.Model:
        """Register RAG model in Vertex AI"""
        logger.info(f"Registering RAG model: {display_name}")
        
        description = (
            f"RAG model for medical Q&A. "
            f"Model: {metadata.get('model_name')}. "
            f"Composite Score: {metadata.get('composite_score', 0):.4f}"
        )
        
        labels = {
            "model-type": "rag",
            "framework": "custom",
            "embedding-model": metadata.get('embedding_model', 'unknown').replace('/', '-').replace('_', '-')
        }
        
        model = aiplatform.Model.upload(
            display_name=display_name,
            artifact_uri=artifact_uri,
            serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/pytorch-cpu.2-1:latest",
            description=description,
            labels=labels
        )
        
        logger.info(f"Model registered: {model.resource_name}")
        return model
    
    def deploy_rag_model(
        self,
        config_path: str,
        index_path: str,
        embeddings_path: str,
        metadata: dict
    ):
        """Complete RAG deployment workflow"""
        logger.info("="*80)
        logger.info("Starting RAG Model Deployment")
        logger.info("="*80)
        
        # Upload artifacts
        uris = self.upload_model_artifacts(
            config_path, index_path, embeddings_path
        )
        
        # Register model
        model = self.register_rag_model(
            display_name="medscan-rag-model",
            artifact_uri=uris['config'],
            metadata=metadata
        )
        
        # Save deployment info
        deployment_info = {
            "model_resource_name": model.resource_name,
            "artifact_uris": uris,
            "metadata": metadata
        }
        
        info_blob = self.bucket.blob("RAG/deployments/latest/deployment_info.json")
        info_blob.upload_from_string(json.dumps(deployment_info, indent=2))
        
        logger.info("="*80)
        logger.info("RAG Deployment Complete!")
        logger.info(f"Model: {model.resource_name}")
        logger.info("="*80)
        
        return model


def main():
    """Main deployment function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Deploy RAG model')
    parser.add_argument('--config', required=True)
    parser.add_argument('--index', required=True)
    parser.add_argument('--embeddings', required=True)
    parser.add_argument('--metadata', required=True)
    
    args = parser.parse_args()
    
    with open(args.metadata, 'r') as f:
        metadata = json.load(f)
    
    deployer = RAGModelDeployer()
    deployer.deploy_rag_model(
        config_path=args.config,
        index_path=args.index,
        embeddings_path=args.embeddings,
        metadata=metadata
    )


if __name__ == "__main__":
    main()