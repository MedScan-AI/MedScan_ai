"""
gcp_utils.py - Common GCP utility functions
"""
import os
import logging
from pathlib import Path
from google.cloud import storage
from google.cloud import aiplatform
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GCPHelper:
    """Helper class for GCP operations"""
    
    def __init__(
        self,
        project_id: str = None,
        bucket_name: str = None,
        region: str = "us-central1"
    ):
        if project_id is None:
            project_id = os.getenv("GCP_PROJECT_ID")
            if not project_id:
                raise ValueError(
                    "GCP_PROJECT_ID not set. Please set it as an environment variable "
                    "or pass it to GCPHelper(project_id='...')"
                )
        if bucket_name is None:
            bucket_name = os.getenv("GCS_BUCKET_NAME")
            if not bucket_name:
                raise ValueError(
                    "GCS_BUCKET_NAME not set. Please set it as an environment variable "
                    "or pass it to GCPHelper(bucket_name='...')"
                )
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.region = region
        
        # Initialize clients
        self.storage_client = storage.Client(project=project_id)
        self.bucket = self.storage_client.bucket(bucket_name)
        
        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=region)
    
    def upload_to_gcs(
        self,
        local_path: str,
        gcs_path: str
    ) -> str:
        """
        Upload file to GCS.
        
        Args:
            local_path: Local file path
            gcs_path: GCS destination path (without gs://bucket/)
            
        Returns:
            Full GCS URI
        """
        blob = self.bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        
        uri = f"gs://{self.bucket_name}/{gcs_path}"
        logger.info(f"Uploaded: {local_path} → {uri}")
        return uri
    
    def download_from_gcs(
        self,
        gcs_path: str,
        local_path: str
    ) -> bool:
        """
        Download file from GCS.
        
        Args:
            gcs_path: GCS source path (without gs://bucket/)
            local_path: Local destination path
            
        Returns:
            True if successful
        """
        try:
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            blob = self.bucket.blob(gcs_path)
            blob.download_to_filename(local_path)
            logger.info(f"Downloaded: gs://{self.bucket_name}/{gcs_path} → {local_path}")
            return True
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False
    
    def list_models(self, filter_str: Optional[str] = None) -> list:
        """List models in Vertex AI Model Registry"""
        return aiplatform.Model.list(filter=filter_str)
    
    def list_endpoints(self, filter_str: Optional[str] = None) -> list:
        """List Vertex AI endpoints"""
        return aiplatform.Endpoint.list(filter=filter_str)


def get_latest_partition_from_gcs(
    bucket_name: str,
    base_path: str
) -> Optional[str]:
    """
    Find latest partition (YYYY/MM/DD) in GCS.
    
    Args:
        bucket_name: GCS bucket name
        base_path: Base path (e.g., 'vision/preprocessed/tb/')
        
    Returns:
        Latest partition path or None
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    # List all blobs with prefix
    blobs = client.list_blobs(bucket_name, prefix=base_path)
    
    # Extract year/month/day patterns
    partitions = set()
    for blob in blobs:
        # Remove base_path to get relative path
        relative = blob.name[len(base_path):].lstrip('/')
        parts = relative.split('/')
        
        # Look for YYYY/MM/DD pattern
        if len(parts) >= 3:
            year, month, day = parts[0:3]
            if year.isdigit() and month.isdigit() and day.isdigit():
                partitions.add(f"{year}/{month}/{day}")
    
    if not partitions:
        return None
    
    # Return latest partition
    latest = max(partitions)
    logger.info(f"Latest partition: {latest}")
    return latest