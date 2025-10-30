"""
URL Manager for RAG Pipeline
Manages URLs list with versioning in GCS
"""
import os
import sys
from pathlib import Path
from datetime import datetime
from google.cloud import storage
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ID = "medscanai-476203"
BUCKET_NAME = "medscan-data"
URLS_PATH = "RAG/config/urls.txt"
VERSIONS_PATH = "RAG/config/versions/"


def get_gcs_client():
    """Get GCS client with credentials."""
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_path:
        creds_path = "/opt/airflow/gcp-service-account.json"
        if not os.path.exists(creds_path):
            creds_path = str(Path.home() / "gcp-service-account.json")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
    
    return storage.Client(project=PROJECT_ID)


def list_versions():
    """List all URL file versions."""
    client = get_gcs_client()
    bucket = client.bucket(BUCKET_NAME)
    
    logger.info("URL File Versions:")
    logger.info("-" * 60)
    
    # Check current version
    current_blob = bucket.blob(URLS_PATH)
    if current_blob.exists():
        current_blob.reload()
        logger.info(f"CURRENT: urls.txt")
        logger.info(f"  Updated: {current_blob.updated}")
        logger.info(f"  Size: {current_blob.size} bytes")
        logger.info("")
    
    # List versioned files
    blobs = list(client.list_blobs(BUCKET_NAME, prefix=VERSIONS_PATH))
    
    if blobs:
        logger.info("Archived Versions:")
        for blob in sorted(blobs, key=lambda x: x.time_created, reverse=True):
            if blob.name.endswith('.txt'):
                logger.info(f"  {blob.name}")
                logger.info(f"    Created: {blob.time_created}")
                logger.info(f"    Size: {blob.size} bytes")
    else:
        logger.info("No archived versions found")


def download_current(output_path=None):
    """Download current URLs file."""
    client = get_gcs_client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(URLS_PATH)
    
    if not blob.exists():
        logger.error("URLs file does not exist in GCS")
        return False
    
    if output_path is None:
        output_path = "urls.txt"
    
    blob.download_to_filename(output_path)
    logger.info(f"Downloaded to: {output_path}")
    
    # Show preview
    with open(output_path, 'r') as f:
        urls = [line.strip() for line in f if line.strip() and line.startswith('http')]
    
    logger.info(f"Total URLs: {len(urls)}")
    logger.info(f"Preview (first 5):")
    for url in urls[:5]:
        logger.info(f"  {url}")
    
    return True


def upload_new(input_path, create_backup=True):
    """Upload new URLs file with optional backup of current version."""
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return False
    
    # Validate file
    with open(input_path, 'r') as f:
        urls = [line.strip() for line in f if line.strip() and line.startswith('http')]
    
    if len(urls) == 0:
        logger.error("No valid URLs found in file")
        return False
    
    logger.info(f"Found {len(urls)} valid URLs")
    
    client = get_gcs_client()
    bucket = client.bucket(BUCKET_NAME)
    current_blob = bucket.blob(URLS_PATH)
    
    # Backup current version
    if create_backup and current_blob.exists():
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = f"{VERSIONS_PATH}urls_{timestamp}.txt"
        backup_blob = bucket.blob(backup_path)
        
        logger.info(f"Backing up current version to: {backup_path}")
        
        # Copy current to backup
        current_content = current_blob.download_as_text()
        backup_blob.upload_from_string(current_content)
        
        logger.info("Backup created")
    
    # Upload new version
    logger.info(f"Uploading new URLs file...")
    current_blob.upload_from_filename(input_path)
    
    logger.info(f"Successfully uploaded {len(urls)} URLs")
    logger.info(f"Location: gs://{BUCKET_NAME}/{URLS_PATH}")
    
    return True


def add_urls(new_urls):
    """Add URLs to existing list."""
    client = get_gcs_client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(URLS_PATH)
    
    # Download current
    if blob.exists():
        current_content = blob.download_as_text()
        existing_urls = set(line.strip() for line in current_content.split('\n') 
                          if line.strip() and line.startswith('http'))
    else:
        existing_urls = set()
    
    logger.info(f"Current URLs: {len(existing_urls)}")
    
    # Add new URLs
    new_urls_list = [url.strip() for url in new_urls if url.strip().startswith('http')]
    added = []
    
    for url in new_urls_list:
        if url not in existing_urls:
            existing_urls.add(url)
            added.append(url)
    
    if not added:
        logger.info("No new URLs to add (all already exist)")
        return True
    
    logger.info(f"Adding {len(added)} new URLs")
    
    # Create backup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = f"{VERSIONS_PATH}urls_{timestamp}.txt"
    backup_blob = bucket.blob(backup_path)
    
    if blob.exists():
        backup_blob.upload_from_string(blob.download_as_text())
        logger.info(f"Backup created: {backup_path}")
    
    # Upload updated list
    updated_content = '\n'.join(sorted(existing_urls)) + '\n'
    blob.upload_from_string(updated_content)
    
    logger.info(f"Updated URLs file: {len(existing_urls)} total URLs")
    logger.info("Added URLs:")
    for url in added:
        logger.info(f"  + {url}")
    
    return True


def restore_version(version_path):
    """Restore a previous version."""
    client = get_gcs_client()
    bucket = client.bucket(BUCKET_NAME)
    
    # Download version
    version_blob = bucket.blob(version_path)
    if not version_blob.exists():
        logger.error(f"Version not found: {version_path}")
        return False
    
    content = version_blob.download_as_text()
    
    # Backup current
    current_blob = bucket.blob(URLS_PATH)
    if current_blob.exists():
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = f"{VERSIONS_PATH}urls_backup_{timestamp}.txt"
        backup_blob = bucket.blob(backup_path)
        backup_blob.upload_from_string(current_blob.download_as_text())
        logger.info(f"Current version backed up to: {backup_path}")
    
    # Restore
    current_blob.upload_from_string(content)
    
    logger.info(f"Restored version: {version_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Manage RAG pipeline URLs')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # List versions
    subparsers.add_parser('list', help='List all URL file versions')
    
    # Download current
    download_parser = subparsers.add_parser('download', help='Download current URLs file')
    download_parser.add_argument('-o', '--output', default='urls.txt', help='Output file path')
    
    # Upload new
    upload_parser = subparsers.add_parser('upload', help='Upload new URLs file')
    upload_parser.add_argument('file', help='Input file path')
    upload_parser.add_argument('--no-backup', action='store_true', help='Skip backup')
    
    # Add URLs
    add_parser = subparsers.add_parser('add', help='Add URLs to existing list')
    add_parser.add_argument('urls', nargs='+', help='URLs to add')
    
    # Restore version
    restore_parser = subparsers.add_parser('restore', help='Restore a previous version')
    restore_parser.add_argument('version', help='Version path (e.g., RAG/config/versions/urls_20250101_120000.txt)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'list':
            list_versions()
        
        elif args.command == 'download':
            download_current(args.output)
        
        elif args.command == 'upload':
            upload_new(args.file, create_backup=not args.no_backup)
        
        elif args.command == 'add':
            add_urls(args.urls)
        
        elif args.command == 'restore':
            restore_version(args.version)
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()