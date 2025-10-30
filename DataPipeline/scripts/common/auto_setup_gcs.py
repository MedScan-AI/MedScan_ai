"""
Automated GCS Bucket Setup for MedScan AI
Creates buckets, folders, and uploads initial configuration
"""
import os
import sys
from pathlib import Path
from google.cloud import storage
from google.api_core import exceptions
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ID = "medscanai-476500"
BUCKET_NAME = "medscan-pipeline-medscanai-476500"

# Medical URLs for RAG pipeline
MEDICAL_URLS = [
    # CDC - Tuberculosis
    "https://www.cdc.gov/tb/treatment/index.html",
    "https://www.cdc.gov/tb/about/index.html",
    "https://www.cdc.gov/tb/signs-symptoms/index.html",
    "https://www.cdc.gov/tb/treatment/active-tuberculosis-disease.html",
    "https://www.cdc.gov/tb/hcp/clinical-overview/index.html",
    "https://www.cdc.gov/tb/hcp/treatment/tuberculosis-disease.html",
    "https://www.cdc.gov/mmwr/volumes/69/rr/rr6901a1.htm",
    "https://www.cdc.gov/infection-control/hcp/core-practices/index.html",
    
    # Mayo Clinic
    "https://www.mayoclinic.org/diseases-conditions/lung-cancer/diagnosis-treatment/drc-20374627",
    "https://www.mayoclinic.org/diseases-conditions/tuberculosis/symptoms-causes/syc-20351250",
    "https://www.mayoclinic.org/diseases-conditions/tuberculosis/diagnosis-treatment/drc-20351256",
    
    # WHO
    "https://www.who.int/news-room/fact-sheets/detail/tuberculosis",
    "https://www.who.int/health-topics/tuberculosis",
    "https://www.who.int/news-room/fact-sheets/detail/cancer",
    
    # Cancer.org
    "https://www.cancer.org/cancer/understanding-cancer/what-is-cancer.html",
    "https://www.cancer.org/cancer/managing-cancer/treatment-types.html",
    "https://www.cancer.org/cancer/managing-cancer/treatment-types/stem-cell-transplant.html",
    "https://www.cancer.org/cancer/managing-cancer/treatment-types/targeted-therapy.html",
    "https://www.cancer.org/cancer/managing-cancer/treatment-types/chemotherapy.html",
    
    # NCI
    "https://www.cancer.gov/about-cancer/treatment/types",
    "https://www.cancer.gov/about-cancer/treatment/types/immunotherapy",
    "https://www.cancer.gov/about-cancer/treatment/types/chemotherapy",
    "https://www.cancer.gov/about-cancer/treatment/types/surgery",
    "https://www.cancer.gov/about-cancer/treatment/types/radiation-therapy",
    "https://www.cancer.gov/about-cancer/treatment/types/targeted-therapies",
    
    # Research Papers
    "https://jamanetwork.com/journals/jama/fullarticle/2777242",
    "https://jamanetwork.com/journals/jama/fullarticle/2804324",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC3876596/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11003524/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC8113854/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC6234945/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC9082420/",
    "https://bmccancer.biomedcentral.com/articles/10.1186/s12885-024-13350-y",
    "https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0161176",
    
    # Cancer Research UK
    "https://www.cancerresearchuk.org/about-cancer/lung-cancer/treatment/non-small-cell-lung-cancer",
    "https://www.cancerresearchuk.org/about-cancer/lung-cancer/treatment/small-cell-lung-cancer",
    "https://www.cancerresearchuk.org/about-cancer/lung-cancer/treatment/chemotherapy-treatment",
    "https://www.cancerresearchuk.org/about-cancer/lung-cancer/treatment/surgery",
    
    # Additional Resources
    "https://www.sciencedirect.com/science/article/pii/S2950162824000195",
    "https://www.mdpi.com/2075-4418/15/7/908",
    "https://www.medicalnewstoday.com/articles/323701",
    "https://go2.org/what-is-lung-cancer/types-of-lung-cancer/",
]


def create_bucket(client, bucket_name, location="us-central1"):
    """Create GCS bucket if it doesn't exist."""
    try:
        bucket = client.get_bucket(bucket_name)
        logger.info(f"Bucket exists: {bucket_name}")
        return bucket
    except exceptions.NotFound:
        logger.info(f"Creating bucket: {bucket_name}")
        bucket = client.create_bucket(bucket_name, location=location)
        logger.info(f"Bucket created: {bucket_name}")
        return bucket


def create_folder_structure(bucket):
    """Create folder structure in GCS bucket."""
    folders = [
        # RAG Pipeline
        "RAG/config/",
        "RAG/config/versions/",
        "RAG/raw_data/baseline/",
        "RAG/raw_data/incremental/",
        "RAG/validation/",
        "RAG/validation/reports/",
        "RAG/chunks/",
        "RAG/embeddings/",
        "RAG/index/",
        
        # Vision Pipeline
        "vision/raw/tb/",
        "vision/raw/lung_cancer/",
        "vision/preprocessed/tb/",
        "vision/preprocessed/lung_cancer/",
        "vision/metadata/tb/",
        "vision/metadata/lung_cancer/",
        "vision/metadata_mitigated/tb/",
        "vision/metadata_mitigated/lung_cancer/",
        "vision/ge_outputs/baseline/",
        "vision/ge_outputs/new_data/",
        "vision/ge_outputs/schemas/",
        "vision/ge_outputs/validations/",
        "vision/ge_outputs/drift/",
        "vision/ge_outputs/bias_analysis/",
        "vision/ge_outputs/eda/",
        "vision/ge_outputs/reports/",
        "vision/mlflow/",
        
        # DVC Storage
        "dvc-storage/vision/",
        "dvc-storage/rag/",
    ]
    
    logger.info("Creating folder structure")
    created = 0
    existing = 0
    
    for folder in folders:
        blob = bucket.blob(folder + ".gitkeep")
        if not blob.exists():
            blob.upload_from_string("")
            created += 1
        else:
            existing += 1
    
    logger.info(f"Folders: {created} created, {existing} existing")
    return len(folders)


def upload_urls_file(bucket):
    """Upload URLs file to GCS if it doesn't exist."""
    blob = bucket.blob("RAG/config/urls.txt")
    
    if blob.exists():
        logger.info(f"URLs file already exists (skipping)")
        return
    
    logger.info(f"Creating URLs file with {len(MEDICAL_URLS)} URLs")
    
    # Create local temp file
    temp_file = Path("/tmp/urls.txt")
    with open(temp_file, 'w') as f:
        for url in MEDICAL_URLS:
            f.write(url + '\n')
    
    # Upload to GCS
    blob.upload_from_filename(str(temp_file))
    
    logger.info(f"Uploaded URLs to gs://{bucket.name}/RAG/config/urls.txt")
    
    # Cleanup
    temp_file.unlink()


def main():
    """Main setup function."""
    logger.info("MedScan AI - Automated GCS Setup")
    
    # Verify credentials
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_path:
        creds_path = "/opt/airflow/gcp-service-account.json"
        if not os.path.exists(creds_path):
            creds_path = str(Path.home() / "gcp-service-account.json")
        
        if os.path.exists(creds_path):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
        else:
            logger.error("GCP credentials not found")
            sys.exit(1)
    
    logger.info(f"Using credentials: {creds_path}")
    
    try:
        # Initialize client
        client = storage.Client(project=PROJECT_ID)
        
        # Create bucket
        bucket = create_bucket(client, BUCKET_NAME)
        
        # Create folder structure
        total_folders = create_folder_structure(bucket)
        
        # Upload URLs file
        upload_urls_file(bucket)
        
        logger.info("GCS Setup Complete")
        logger.info(f"Bucket: gs://{BUCKET_NAME}")
        logger.info(f"Total Folders: {total_folders}")
        logger.info(f"URLs: {len(MEDICAL_URLS)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Setup failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()