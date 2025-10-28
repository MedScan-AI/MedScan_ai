"""
URL Manager for RAG Pipeline
"""
import logging
from pathlib import Path

# Import unified config
from DataPipeline.config import gcp_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_urls_from_gcs(gcs_manager, gcs_path: str) -> list:
    """Read URLs from GCS configuration file."""
    logger.info(f"Reading URLs from GCS: {gcs_path}")
    
    # Download to temp location
    local_temp = gcp_config.LOCAL_PATHS['temp'] / "urls.txt"
    local_temp.parent.mkdir(parents=True, exist_ok=True)
    
    success = gcs_manager.download_file(gcs_path, str(local_temp))
    
    if not success:
        logger.warning(f"URLs file not found in GCS: {gcs_path}")
        logger.info("Using default URLs")
        return get_default_urls()
    
    # Read URLs
    urls = []
    with open(local_temp, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and line.startswith('http'):
                urls.append(line)
    
    logger.info(f"Loaded {len(urls)} URLs from GCS")
    return urls


def get_default_urls() -> list:
    """Get default URLs if GCS file doesn't exist."""
    return [
        "https://www.cdc.gov/tb/treatment/index.html",
        "https://www.mayoclinic.org/diseases-conditions/lung-cancer/diagnosis-treatment/drc-20374627",
        "https://www.cancer.org/cancer/understanding-cancer/what-is-cancer.html",
        "https://www.who.int/news-room/fact-sheets/detail/tuberculosis",
    ]


def upload_urls_to_gcs(gcs_manager, urls: list, gcs_path: str):
    """Upload URLs to GCS."""
    local_temp = gcp_config.LOCAL_PATHS['temp'] / "urls_upload.txt"
    local_temp.parent.mkdir(parents=True, exist_ok=True)
    
    # Write URLs
    with open(local_temp, 'w') as f:
        for url in urls:
            f.write(url + '\n')
    
    # Upload
    gcs_manager.upload_file(str(local_temp), gcs_path)
    logger.info(f"Uploaded {len(urls)} URLs to GCS: {gcs_path}")