"""GCS operations for RAG pipeline."""
import logging
from pathlib import Path
from google.cloud import storage
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GCSManager:
    """Manage GCS operations."""
    
    def __init__(self, bucket_name: str, credentials_path: str = None):
        if credentials_path and os.path.exists(credentials_path):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.bucket_name = bucket_name
        logger.info(f"GCSManager: {bucket_name}")
    
    def upload_file(self, local_path: str, gcs_path: str) -> bool:
        """Upload file to GCS."""
        try:
            blob = self.bucket.blob(gcs_path)
            blob.upload_from_filename(local_path)
            size = os.path.getsize(local_path)
            logger.info(f"Uploaded: {gcs_path} ({size:,} bytes)")
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
                logger.warning(f"Not found: {gcs_path}")
                return False
            
            blob.download_to_filename(local_path)
            logger.info(f"Downloaded: {gcs_path}")
            return True
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False
    
    def blob_exists(self, gcs_path: str) -> bool:
        """Check if blob exists."""
        try:
            return self.bucket.blob(gcs_path).exists()
        except:
            return False
    
    def list_blobs(self, prefix: str) -> list:
        """List blobs with prefix."""
        try:
            blobs = self.client.list_blobs(self.bucket_name, prefix=prefix)
            return [blob.name for blob in blobs]
        except:
            return []
    
    def get_latest_version(self, prefix: str, pattern: str = "v") -> int:
        """Get latest version number."""
        try:
            blobs = self.list_blobs(prefix)
            versions = []
            
            for blob_name in blobs:
                if pattern in blob_name and not blob_name.endswith('.gitkeep'):
                    try:
                        version_str = blob_name.split(pattern)[-1].split(".")[0]
                        versions.append(int(version_str))
                    except:
                        continue
            
            return max(versions) if versions else 0
        except:
            return 0


def upload_with_versioning(gcs_manager, local_file, gcs_prefix, filename_template, also_latest=True):
    """Upload with auto-versioning."""
    current_ver = gcs_manager.get_latest_version(gcs_prefix)
    next_ver = current_ver + 1
    
    versioned = filename_template.replace("{version}", f"v{next_ver:03d}")
    gcs_manager.upload_file(local_file, f"{gcs_prefix}/{versioned}")
    
    if also_latest:
        latest = filename_template.replace("{version}", "latest")
        gcs_manager.upload_file(local_file, f"{gcs_prefix}/{latest}")
    
    logger.info(f"Version {next_ver} uploaded")
    return next_ver