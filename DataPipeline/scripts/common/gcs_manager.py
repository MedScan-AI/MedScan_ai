"""GCS Manager for MedScan AI"""
import logging
import os
import socket
import time
from pathlib import Path
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from google.cloud import storage
from google.api_core.exceptions import GoogleAPIError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GCSManager:
    """
    GCS operations manager for all MedScan pipelines.
    """
    
    def __init__(
        self, 
        bucket_name: str, 
        credentials_path: Optional[str] = None,
        project_id: Optional[str] = None
    ):
        """
        Initialize GCS Manager.
        
        Args:
            bucket_name: GCS bucket name
            credentials_path: Path to service account JSON (optional)
            project_id: GCP project ID (optional)
        """
        if credentials_path and os.path.exists(credentials_path):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        
        self.project_id = project_id
        self.client = storage.Client(project=project_id) if project_id else storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.bucket_name = bucket_name
        
        logger.info(f"GCSManager initialized for bucket: {bucket_name}")
    
    @classmethod
    def from_config(cls):
        """Create GCS manager from config."""
        from DataPipeline.config import gcp_config
        return cls(
            bucket_name=gcp_config.BUCKET_NAME,
            credentials_path=str(gcp_config.SERVICE_ACCOUNT_PATH),
            project_id=gcp_config.PROJECT_ID
        )
    
    # Core Operations
    def blob_exists(self, gcs_path: str, timeout: int = 10) -> bool:
        """Check if blob exists with timeout."""
        start = time.time()
        try:
            blob = self.bucket.blob(gcs_path)
            socket.setdefaulttimeout(timeout)
            exists = blob.exists(timeout=timeout)
            elapsed = time.time() - start
            logger.debug(f"[{elapsed:.2f}s] exists({gcs_path}) → {exists}")
            return exists
        except GoogleAPIError as e:
            logger.error(f"GCS API error: {e}")
            return False
        except socket.timeout:
            logger.error(f"Timeout ({timeout}s) checking {gcs_path}")
            return False
        except Exception as e:
            logger.error(f"Error checking {gcs_path}: {e}")
            return False
        finally:
            socket.setdefaulttimeout(None)
    
    def download_file(self, gcs_path: str, local_path: str, timeout: int = 30) -> bool:
        """Download file from GCS."""
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
            logger.error(f"Timeout downloading {gcs_path}")
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
            logger.error(f"Upload failed {gcs_path}: {e}")
            return False
    
    def list_blobs(self, prefix: str) -> List[str]:
        """List blobs with prefix."""
        try:
            blobs = self.client.list_blobs(self.bucket_name, prefix=prefix)
            return [blob.name for blob in blobs]
        except Exception as e:
            logger.error(f"List failed for prefix={prefix}: {e}")
            return []
    
    def create_marker(self, gcs_path: str, content: str = "") -> bool:
        """Create a marker file in GCS."""
        try:
            from datetime import datetime
            blob = self.bucket.blob(gcs_path)
            blob.upload_from_string(content or f"Completed at {datetime.utcnow().isoformat()}")
            logger.info(f" Created marker: {gcs_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create marker {gcs_path}: {e}")
            return False
    
    
    # Directory Operations (Parallel)
    
    
    def _upload_single_file(self, args: Tuple[str, str]) -> Tuple[bool, str]:
        """Helper for parallel upload."""
        local_path, gcs_path = args
        success = self.upload_file(local_path, gcs_path)
        return (success, gcs_path)
    
    def upload_directory(
        self, 
        local_dir: str, 
        gcs_prefix: str, 
        max_workers: int = 10,
        preserve_structure: bool = True
    ) -> int:
        """
        Upload directory to GCS with parallel uploads.
        
        Args:
            local_dir: Local directory path
            gcs_prefix: GCS prefix (without trailing slash)
            max_workers: Number of parallel threads
            preserve_structure: Preserve directory structure
        
        Returns:
            Number of files uploaded
        """
        local_path = Path(local_dir)
        
        if not local_path.exists():
            logger.error(f"Directory not found: {local_dir}")
            return 0
        
        gcs_prefix = gcs_prefix.rstrip('/')
        
        # Collect files to upload
        upload_tasks: List[Tuple[str, str]] = []
        for file_path in local_path.rglob('*'):
            if file_path.is_file():
                if preserve_structure:
                    relative_path = file_path.relative_to(local_path)
                    gcs_path = f"{gcs_prefix}/{relative_path}"
                else:
                    gcs_path = f"{gcs_prefix}/{file_path.name}"
                upload_tasks.append((str(file_path), gcs_path))
        
        if not upload_tasks:
            logger.warning(f"No files in {local_dir}")
            return 0
        
        logger.info(f"Uploading {len(upload_tasks)} files with {max_workers} workers...")
        
        # Parallel upload
        success_count = 0
        failed_files = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._upload_single_file, task): task for task in upload_tasks}
            
            for future in as_completed(futures):
                try:
                    success, gcs_path = future.result()
                    if success:
                        success_count += 1
                    else:
                        failed_files.append(gcs_path)
                except Exception as e:
                    task = futures[future]
                    logger.error(f"Upload exception for {task[1]}: {e}")
                    failed_files.append(task[1])
        
        if failed_files:
            logger.warning(f"Failed: {len(failed_files)} files")
            for failed in failed_files[:10]:
                logger.warning(f"  - {failed}")
        
        logger.info(f" Uploaded {success_count}/{len(upload_tasks)} files")
        return success_count
    
    def _download_single_blob(self, args: Tuple[storage.Blob, str, str]) -> bool:
        """Helper for parallel download."""
        blob, gcs_prefix, local_dir = args
        try:
            relative_path = blob.name[len(gcs_prefix):].lstrip('/')
            local_path = Path(local_dir) / relative_path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(local_path))
            return True
        except Exception as e:
            logger.error(f"Download failed {blob.name}: {e}")
            return False
    
    def download_directory(
        self, 
        gcs_prefix: str, 
        local_dir: str, 
        max_workers: int = 10
    ) -> int:
        """Download directory from GCS with parallel downloads."""
        gcs_prefix = gcs_prefix.rstrip('/')
        
        try:
            blobs = list(self.client.list_blobs(self.bucket_name, prefix=gcs_prefix))
            file_blobs = [b for b in blobs if not b.name.endswith('/')]
            
            if not file_blobs:
                logger.warning(f"No files with prefix: {gcs_prefix}")
                return 0
            
            logger.info(f"Downloading {len(file_blobs)} files with {max_workers} workers...")
            
            # Parallel download
            success_count = 0
            download_tasks = [(blob, gcs_prefix, local_dir) for blob in file_blobs]
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = executor.map(self._download_single_blob, download_tasks)
                success_count = sum(1 for r in results if r)
            
            logger.info(f" Downloaded {success_count}/{len(file_blobs)} files")
            return success_count
            
        except Exception as e:
            logger.error(f"Download directory failed: {e}")
            return 0
    
    
    # Versioning Support
    def get_latest_version(self, prefix: str, pattern: str = "v") -> int:
        """Get latest version number from blob names."""
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
        self,
        local_file: str,
        gcs_prefix: str,
        filename_template: str,
        also_latest: bool = True
    ) -> int:
        """
        Upload file with automatic versioning.
        
        Args:
            local_file: Local file path
            gcs_prefix: GCS folder/prefix
            filename_template: Filename with {version} placeholder
            also_latest: Also upload as 'latest'
        
        Returns:
            Next version number
        """
        current_version = self.get_latest_version(gcs_prefix)
        next_version = current_version + 1
        
        # Upload versioned
        versioned_filename = filename_template.replace("{version}", f"{next_version:03d}")
        versioned_path = f"{gcs_prefix}/{versioned_filename}"
        self.upload_file(local_file, versioned_path)
        
        # Upload latest
        if also_latest:
            latest_filename = filename_template.replace("{version}", "latest")
            latest_path = f"{gcs_prefix}/{latest_filename}"
            self.upload_file(local_file, latest_path)
        
        return next_version