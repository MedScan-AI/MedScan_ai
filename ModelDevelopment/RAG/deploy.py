"""
deploy.py - Deploy RAG model to Vertex AI
Supports both local files and GCS URIs
"""
import logging
import json
from pathlib import Path
from google.cloud import aiplatform
from google.cloud import storage
import tempfile

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
    
    def _resolve_path(self, path: str) -> str:
        """
        Resolve path to GCS URI.
        If already GCS URI, return as-is.
        If local path, upload to GCS and return URI.
        
        Args:
            path: Local path or GCS URI
            
        Returns:
            GCS URI
        """
        # Already a GCS URI
        if path.startswith('gs://'):
            return path
        
        # Local file - upload to GCS
        local_path = Path(path)
        if not local_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        # Upload to temporary location in GCS
        filename = local_path.name
        gcs_path = f"RAG/models/uploads/{filename}"
        
        logger.info(f"Uploading {filename} to GCS...")
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
    ) -> dict:
        """
        Upload RAG model artifacts to GCS (or use existing GCS URIs).
        
        Args:
            config_path: Path to config.json (local or GCS URI)
            index_path: Path to index.bin (local or GCS URI)
            embeddings_path: Path to embeddings.json (local or GCS URI)
        
        Returns:
            Dict with GCS URIs
        """
        logger.info("Processing RAG model artifacts...")
        
        uris = {}
        
        # Resolve config
        config_uri = self._resolve_path(config_path)
        
        # Copy to latest location
        if config_uri.startswith('gs://'):
            # Parse source URI
            source_parts = config_uri.replace('gs://', '').split('/', 1)
            source_bucket = source_parts[0]
            source_blob_path = source_parts[1]
            
            # Copy to latest
            dest_blob = self.bucket.blob("RAG/models/latest/config.json")
            
            if source_bucket == self.bucket_name:
                # Same bucket - copy blob
                source_blob = self.bucket.blob(source_blob_path)
                self.bucket.copy_blob(source_blob, self.bucket, dest_blob.name)
            else:
                # Different bucket - download and re-upload
                source_client = storage.Client()
                source_bucket_obj = source_client.bucket(source_bucket)
                source_blob = source_bucket_obj.blob(source_blob_path)
                
                content = source_blob.download_as_text()
                dest_blob.upload_from_string(content)
        
        uris['config'] = f"gs://{self.bucket_name}/RAG/models/latest/config.json"
        
        # Resolve and copy index
        index_uri = self._resolve_path(index_path)
        if index_uri.startswith('gs://'):
            source_parts = index_uri.replace('gs://', '').split('/', 1)
            source_blob_path = source_parts[1]
            source_blob = self.bucket.blob(source_blob_path)
            dest_blob = self.bucket.blob("RAG/models/latest/index.bin")
            self.bucket.copy_blob(source_blob, self.bucket, dest_blob.name)
        
        uris['index'] = f"gs://{self.bucket_name}/RAG/models/latest/index.bin"
        
        # Resolve and copy embeddings
        embeddings_uri = self._resolve_path(embeddings_path)
        if embeddings_uri.startswith('gs://'):
            source_parts = embeddings_uri.replace('gs://', '').split('/', 1)
            source_blob_path = source_parts[1]
            source_blob = self.bucket.blob(source_blob_path)
            dest_blob = self.bucket.blob("RAG/models/latest/embeddings.json")
            self.bucket.copy_blob(source_blob, self.bucket, dest_blob.name)
        
        uris['embeddings'] = f"gs://{self.bucket_name}/RAG/models/latest/embeddings.json"
        
        logger.info("Artifacts processed successfully")
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
            f"Model: {metadata.get('model_name', 'unknown')}. "
            f"Composite Score: {metadata.get('performance_metrics', {}).get('composite_score', 0):.4f}"
        )
        
        # Clean labels (replace invalid characters)
        embedding_model = metadata.get('embedding_model', 'unknown').replace('/', '-').replace('_', '-').lower()
        
        labels = {
            "model-type": "rag",
            "framework": "custom",
            "embedding-model": embedding_model[:63]  # GCP label limit
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
        metadata_path: str
    ):
        """
        Complete RAG deployment workflow.
        
        Args:
            config_path: Path to config.json (local or GCS URI)
            index_path: Path to index.bin (local or GCS URI)
            embeddings_path: Path to embeddings.json (local or GCS URI)
            metadata_path: Path to metadata JSON
        """
        logger.info("="*80)
        logger.info("🚀 Starting RAG Model Deployment")
        logger.info("="*80)
        
        # Load metadata
        if metadata_path.startswith('gs://'):
            # Download from GCS
            parts = metadata_path.replace('gs://', '').split('/', 1)
            bucket = self.storage_client.bucket(parts[0])
            blob = bucket.blob(parts[1])
            metadata = json.loads(blob.download_as_text())
        else:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        # Upload artifacts to standard locations
        uris = self.upload_model_artifacts(
            config_path, index_path, embeddings_path
        )
        
        # Register model
        model = self.register_rag_model(
            display_name="medscan-rag-model",
            artifact_uri=uris['config'],  # Use config URI as main artifact
            metadata=metadata
        )
        
        # Save deployment info
        deployment_info = {
            "model_resource_name": model.resource_name,
            "artifact_uris": uris,
            "metadata": metadata,
            "deployment_timestamp": aiplatform.utils.get_timestamp()
        }
        
        info_blob = self.bucket.blob("RAG/deployments/latest/deployment_info.json")
        info_blob.upload_from_string(json.dumps(deployment_info, indent=2))
        
        logger.info("="*80)
        logger.info("RAG Deployment Complete!")
        logger.info(f"Model: {model.resource_name}")
        logger.info(f"Config: {uris['config']}")
        logger.info(f"Index: {uris['index']}")
        logger.info(f"Embeddings: {uris['embeddings']}")
        logger.info("="*80)
        
        return model


def main():
    """Main deployment function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Deploy RAG model to Vertex AI')
    parser.add_argument('--config', required=True, help='Path to config.json (local or gs:// URI)')
    parser.add_argument('--index', required=True, help='Path to index.bin (local or gs:// URI)')
    parser.add_argument('--embeddings', required=True, help='Path to embeddings.json (local or gs:// URI)')
    parser.add_argument('--metadata', required=True, help='Path to metadata JSON')
    
    args = parser.parse_args()
    
    deployer = RAGModelDeployer()
    deployer.deploy_rag_model(
        config_path=args.config,
        index_path=args.index,
        embeddings_path=args.embeddings,
        metadata_path=args.metadata
    )


if __name__ == "__main__":
    main()