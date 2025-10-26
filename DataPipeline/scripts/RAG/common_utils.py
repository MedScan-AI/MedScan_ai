"""Common utilities for RAG pipeline."""
import logging
from pathlib import Path
from google.cloud import storage
import os

logger = logging.getLogger(__name__)


class GCSManager:
    """Manage GCS operations for RAG pipeline."""
    
    def __init__(self, bucket_name: str, credentials_path: str = None):
        if credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.bucket_name = bucket_name
    
    def upload_file(self, local_path: str, gcs_path: str) -> bool:
        """Upload file to GCS."""
        try:
            blob = self.bucket.blob(gcs_path)
            blob.upload_from_filename(local_path)
            logger.info(f"Uploaded {local_path} → gs://{self.bucket_name}/{gcs_path}")
            return True
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return False
    
    def download_file(self, gcs_path: str, local_path: str) -> bool:
        """Download file from GCS."""
        try:
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            blob = self.bucket.blob(gcs_path)
            
            if not blob.exists():
                logger.warning(f"File not found: {gcs_path}")
                return False
            
            blob.download_to_filename(local_path)
            logger.info(f"Downloaded gs://{self.bucket_name}/{gcs_path} → {local_path}")
            return True
        except Exception as e:
            logger.error(f" Download failed: {e}")
            return False
    
    def blob_exists(self, gcs_path: str) -> bool:
        """Check if blob exists."""
        try:
            blob = self.bucket.blob(gcs_path)
            return blob.exists()
        except:
            return False
    
    def get_latest_version(self, prefix: str, pattern: str = "v") -> int:
        """Get latest version number from GCS."""
        try:
            blobs = list(self.client.list_blobs(self.bucket_name, prefix=prefix))
            versions = []
            
            for blob in blobs:
                if pattern in blob.name:
                    try:
                        version_str = blob.name.split(pattern)[-1].split(".")[0]
                        versions.append(int(version_str))
                    except:
                        continue
            
            return max(versions) if versions else 0
        except:
            return 0


def upload_with_versioning(
    gcs_manager: GCSManager,
    local_file: str,
    gcs_prefix: str,
    filename_template: str,
    also_latest: bool = True
) -> int:
    """Upload file with automatic versioning."""
    current_version = gcs_manager.get_latest_version(gcs_prefix)
    next_version = current_version + 1
    
    # Upload versioned
    versioned_filename = filename_template.replace("{version}", f"{next_version:03d}")
    versioned_path = f"{gcs_prefix}/{versioned_filename}"
    gcs_manager.upload_file(local_file, versioned_path)
    
    # Upload latest
    if also_latest:
        latest_filename = filename_template.replace("{version}", "latest")
        latest_path = f"{gcs_prefix}/{latest_filename}"
        gcs_manager.upload_file(local_file, latest_path)
    
    return next_version