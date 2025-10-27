"""Common utilities for RAG pipeline."""
import logging
from pathlib import Path
from google.cloud import storage
from google.api_core.exceptions import GoogleAPIError
import os
import socket
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GCSManager:
    """Manage GCS operations for RAG pipeline."""

    def __init__(self, bucket_name: str, credentials_path: str = None):
        if credentials_path and os.path.exists(credentials_path):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.bucket_name = bucket_name
        logger.info(f"GCSManager initialized for bucket: {bucket_name}")


    def blob_exists(self, gcs_path: str, timeout: int = 10) -> bool:
        """Check if blob exists with timeout + error handling."""
        start = time.time()
        try:
            blob = self.bucket.blob(gcs_path)
            socket.setdefaulttimeout(timeout)
            exists = blob.exists(timeout=timeout)
            elapsed = time.time() - start
            logger.info(f"[{elapsed:.2f}s] exists({gcs_path}) → {exists}")
            return exists
        except GoogleAPIError as e:
            logger.error(f"GCS API error checking existence: {e}")
            return False
        except socket.timeout:
            logger.error(f"Timeout ({timeout}s) checking existence of {gcs_path}")
            return False
        except Exception as e:
            logger.error(f"Unknown error checking {gcs_path}: {e}")
            return False
        finally:
            socket.setdefaulttimeout(None)


    def download_file(self, gcs_path: str, local_path: str, timeout: int = 30) -> bool:
        """Download file from GCS with timeout."""
        try:
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            blob = self.bucket.blob(gcs_path)

            if not self.blob_exists(gcs_path, timeout=timeout):
                logger.warning(f"Not found: {gcs_path}")
                return False

            socket.setdefaulttimeout(timeout)
            blob.download_to_filename(local_path, timeout=timeout)
            logger.info(f"Downloaded: {gcs_path} → {local_path}")
            return True
        except socket.timeout:
            logger.error(f"Timeout ({timeout}s) downloading {gcs_path}")
            return False
        except GoogleAPIError as e:
            logger.error(f"GCS download error: {e}")
            return False
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False
        finally:
            socket.setdefaulttimeout(None)


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


    def list_blobs(self, prefix: str) -> list:
        """List blobs with prefix."""
        try:
            blobs = self.client.list_blobs(self.bucket_name, prefix=prefix)
            return [blob.name for blob in blobs]
        except Exception as e:
            logger.error(f"List failed for prefix={prefix}: {e}")
            return []


    def get_latest_version(self, prefix: str, pattern: str = "v") -> int:
        """Get latest version number."""
        try:
            blobs = self.list_blobs(prefix)
            versions = []
            for blob_name in blobs:
                if pattern in blob_name and not blob_name.endswith(".gitkeep"):
                    try:
                        version_str = blob_name.split(pattern)[-1].split(".")[0]
                        versions.append(int(version_str))
                    except Exception:
                        continue
            return max(versions) if versions else 0
        except Exception as e:
            logger.error(f"Version detection failed: {e}")
            return 0


def upload_with_versioning(
    gcs_manager: GCSManager,
    local_file: str,
    gcs_prefix: str,
    filename_template: str,
    also_latest: bool = True
) -> int:
    """
    Upload file with automatic versioning.

    Args:
        gcs_manager: GCSManager instance
        local_file: local file path to upload
        gcs_prefix: GCS folder/prefix to upload to
        filename_template: filename with {version} placeholder
        also_latest: if True, also upload a copy as 'latest'

    Returns:
        next version number (int)
    """
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