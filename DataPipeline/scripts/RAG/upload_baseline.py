"""Upload baseline data to GCS as FIXED reference baseline."""
import os
import sys
from pathlib import Path
from google.cloud import storage

# Paths
BASELINE_FILE = Path(__file__).parent.parent.parent.parent / "data" / "RAG" / "raw_data" / "baseline.jsonl"
SERVICE_ACCOUNT = Path.home() / "gcp-service-account.json"
BUCKET_NAME = "medscan-rag-data"
GCS_PATH = "RAG/raw_data/baseline/baseline.jsonl"
PROJECT_ID = "medscanai-476203"


def upload_baseline():
    """Upload baseline to GCS as FIXED reference.
    
    This creates the FIXED baseline that all future runs will compare against.
    """
    print("=" * 70)
    print("UPLOADING FIXED BASELINE TO GCS")
    print("=" * 70)
    print()
    
    # Validate local file exists
    if not BASELINE_FILE.exists():
        print(f"Baseline file not found: {BASELINE_FILE}")
        print()
        print("Create it first:")
        print("  cd DataPipeline/scripts/RAG")
        print("  python scraper.py \\")
        print("    -u [urls] \\")
        print("    -o ../../../data/RAG/raw_data/baseline.jsonl \\")
        print("    -m W")
        sys.exit(1)
    
    # Check file size
    file_size = BASELINE_FILE.stat().st_size
    print(f"Local File:")
    print(f"   Path: {BASELINE_FILE}")
    print(f"   Size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
    
    # Count records
    with open(BASELINE_FILE, 'r') as f:
        record_count = sum(1 for _ in f)
    print(f"   Records: {record_count}")
    print()
    
    # Validate credentials
    if not SERVICE_ACCOUNT.exists():
        print(f"Service account not found: {SERVICE_ACCOUNT}")
        print()
        print("Download from GCP Console:")
        print("  1. Go to: https://console.cloud.google.com")
        print("  2. IAM & Admin → Service Accounts")
        print("  3. Create/select service account")
        print("  4. Keys → Add Key → Create New Key (JSON)")
        print(f"  5. Save as: {SERVICE_ACCOUNT}")
        sys.exit(1)
    
    # Set credentials
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(SERVICE_ACCOUNT)
    
    try:
        # Connect to GCS
        print(f"Authenticating with GCP...")
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(BUCKET_NAME)
        print(f"Connected to project: {PROJECT_ID}")
        print(f"Using bucket: {BUCKET_NAME}")
        print()
        
        # Check if baseline already exists
        blob = bucket.blob(GCS_PATH)
        
        if blob.exists():
            print("WARNING: FIXED BASELINE ALREADY EXISTS!")
            print("=" * 70)
            print(f"   Location: gs://{BUCKET_NAME}/{GCS_PATH}")
            print(f"   Size: {blob.size:,} bytes")
            print(f"   Created: {blob.time_created}")
            print()
            print("   Overwriting will change the FIXED baseline that all")
            print("   future validations compare against!")
            print("=" * 70)
            print()
            
            response = input("Overwrite FIXED baseline? (type 'yes' to confirm): ")
            if response.lower() != 'yes':
                print()
                print("Upload cancelled - baseline unchanged")
                sys.exit(0)
            print()
        
        # Upload baseline
        print(f"Uploading to GCS...")
        blob.upload_from_filename(str(BASELINE_FILE))
        
        # Verify upload
        blob.reload()
        
        print()
        print("=" * 70)
        print("✓ FIXED BASELINE UPLOADED SUCCESSFULLY!")
        print("=" * 70)
        print(f"   Local:  {BASELINE_FILE}")
        print(f"   GCS:    gs://{BUCKET_NAME}/{GCS_PATH}")
        print(f"   Size:   {blob.size:,} bytes")
        print(f"   Records: {record_count}")
        print(f"   MD5:    {blob.md5_hash}")
        print()
        print("IMPORTANT:")
        print("   • This is your FIXED BASELINE")
        print("   • All future pipeline runs will compare against THIS data")
        print("   • TFDV will detect drift from this reference point")
        print("   • Only update this baseline manually when:")
        print("     - Data standards change")
        print("     - Medical guidelines update")
        print("     - Model retraining requires new reference")
        print("=" * 70)
        
    except Exception as e:
        print()
        print(f"Upload failed: {e}")
        print()
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    upload_baseline()