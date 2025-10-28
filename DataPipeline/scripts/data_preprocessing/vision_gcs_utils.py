"""GCS utilities for vision pipeline (mirrors RAG/common_utils.py)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "config"))

import vision_gcp_config
from google.cloud import storage
import logging
import os
import time
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

logger = logging.getLogger(__name__)


class VisionGCSManager:
    """Manage GCS operations for vision pipeline."""
    
    def __init__(self):
        creds_path = vision_gcp_config.SERVICE_ACCOUNT_PATH
        if creds_path and os.path.exists(creds_path):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds_path)
        
        self.client = storage.Client(project=vision_gcp_config.GCP_PROJECT_ID)
        self.bucket = self.client.bucket(vision_gcp_config.GCS_BUCKET_NAME)
        self.bucket_name = vision_gcp_config.GCS_BUCKET_NAME
        logger.info(f"VisionGCSManager initialized: {self.bucket_name}")
    
    def blob_exists(self, gcs_path: str, timeout: int = 10) -> bool:
        """Check if blob exists."""
        try:
            blob = self.bucket.blob(gcs_path)
            socket.setdefaulttimeout(timeout)
            exists = blob.exists(timeout=timeout)
            logger.debug(f"Exists: {gcs_path} → {exists}")
            return exists
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
            logger.debug(f"Uploaded: {gcs_path} ({size:,} bytes)")
            return True
        except Exception as e:
            logger.error(f"Upload failed {gcs_path}: {e}")
            return False
    
    def _upload_single_file(self, args: Tuple[str, str]) -> Tuple[bool, str]:
        """Helper for parallel upload. Returns (success, gcs_path)."""
        local_path, gcs_path = args
        success = self.upload_file(local_path, gcs_path)
        return (success, gcs_path)
    
    def upload_directory(self, local_dir: str, gcs_prefix: str, max_workers: int = 10) -> int:
        """
        Upload entire directory to GCS preserving structure (WITH PARALLEL UPLOADS).
        
        Args:
            local_dir: Local directory path (e.g., '/opt/airflow/DataPipeline/data/raw/tb')
            gcs_prefix: GCS prefix WITHOUT trailing slash (e.g., 'vision/raw/tb/2025/10/28')
            max_workers: Number of parallel upload threads (default: 10)
        
        Returns:
            Number of files successfully uploaded
        """
        local_path = Path(local_dir)
        
        if not local_path.exists():
            logger.error(f"Local directory does not exist: {local_dir}")
            return 0
        
        # Remove trailing slash from gcs_prefix if present
        gcs_prefix = gcs_prefix.rstrip('/')
        
        # Collect all files to upload
        upload_tasks: List[Tuple[str, str]] = []
        for file_path in local_path.rglob('*'):
            if file_path.is_file():
                relative_path = file_path.relative_to(local_path)
                # FIX: Add slash between prefix and relative path
                gcs_path = f"{gcs_prefix}/{relative_path}"
                upload_tasks.append((str(file_path), gcs_path))
        
        if not upload_tasks:
            logger.warning(f"No files found in {local_dir}")
            return 0
        
        logger.info(f"Uploading {len(upload_tasks)} files using {max_workers} workers...")
        
        # Upload in parallel
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
            logger.warning(f"Failed to upload {len(failed_files)} files")
            for failed in failed_files[:10]:  # Log first 10 failures
                logger.warning(f"  - {failed}")
        
        logger.info(f"✓ Uploaded {success_count}/{len(upload_tasks)} files from {local_dir} to gs://{self.bucket_name}/{gcs_prefix}/")
        return success_count
    
    def download_directory(self, gcs_prefix: str, local_dir: str, max_workers: int = 10) -> int:
        """
        Download all files with prefix to local directory (WITH PARALLEL DOWNLOADS).
        
        Args:
            gcs_prefix: GCS prefix (e.g., 'vision/raw/tb/2025/10/28')
            local_dir: Local directory to download to
            max_workers: Number of parallel download threads
        
        Returns:
            Number of files downloaded
        """
        gcs_prefix = gcs_prefix.rstrip('/')
        
        try:
            # List all blobs
            blobs = list(self.client.list_blobs(self.bucket_name, prefix=gcs_prefix))
            
            # Filter out directory markers
            file_blobs = [b for b in blobs if not b.name.endswith('/')]
            
            if not file_blobs:
                logger.warning(f"No files found with prefix: {gcs_prefix}")
                return 0
            
            logger.info(f"Downloading {len(file_blobs)} files using {max_workers} workers...")
            
            def download_single(blob):
                try:
                    relative_path = blob.name[len(gcs_prefix):].lstrip('/')
                    local_path = Path(local_dir) / relative_path
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    blob.download_to_filename(str(local_path))
                    return True
                except Exception as e:
                    logger.error(f"Download failed {blob.name}: {e}")
                    return False
            
            # Download in parallel
            success_count = 0
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = executor.map(download_single, file_blobs)
                success_count = sum(1 for r in results if r)
            
            logger.info(f"✓ Downloaded {success_count}/{len(file_blobs)} files from gs://{self.bucket_name}/{gcs_prefix}/ to {local_dir}")
            return success_count
            
        except Exception as e:
            logger.error(f"Download directory failed: {e}")
            return 0
    
    def create_marker(self, gcs_path: str, content: str = "") -> bool:
        """Create a marker file in GCS."""
        try:
            from datetime import datetime
            blob = self.bucket.blob(gcs_path)
            blob.upload_from_string(content or f"Completed at {datetime.utcnow().isoformat()}")
            logger.info(f"✓ Created marker: {gcs_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create marker {gcs_path}: {e}")
            return False
    
    def list_blobs(self, prefix: str) -> list:
        """List blobs with prefix."""
        try:
            blobs = self.client.list_blobs(self.bucket_name, prefix=prefix)
            return [blob.name for blob in blobs]
        except Exception as e:
            logger.error(f"List failed: {e}")
            return []