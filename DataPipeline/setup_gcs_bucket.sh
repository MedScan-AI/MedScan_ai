#!/bin/bash
set -e

echo "=========================================="
echo "One-Time GCS Setup for RAG Pipeline"
echo "=========================================="
echo ""

# Check credentials
if [ ! -f ~/gcp-service-account.json ]; then
    echo "Error: ~/gcp-service-account.json not found"
    echo ""
    echo "Please:"
    echo "  1. Download service account JSON from GCP Console"
    echo "  2. Save as: ~/gcp-service-account.json"
    echo "  3. Run this script again"
    exit 1
fi

echo " GCP credentials found"
echo ""

# Set credentials
export GOOGLE_APPLICATION_CREDENTIALS=~/gcp-service-account.json

# Create bucket and folders
python3 << 'EOF'
from google.cloud import storage

print("Creating GCS bucket and folders...")
print()

try:
    client = storage.Client(project="medscanai-476203")
    bucket_name = "medscan-rag-data"
    
    # Create bucket
    try:
        bucket = client.get_bucket(bucket_name)
        print(f" Bucket already exists: {bucket_name}")
    except:
        bucket = client.create_bucket(bucket_name, location="us-central1")
        print(f" Created bucket: {bucket_name}")
    
    print()
    
    # Create folders
    folders = [
        "RAG/raw_data/baseline/",
        "RAG/raw_data/incremental/",
        "RAG/merged/",
        "RAG/validation/baseline/",
        "RAG/validation/runs/",
        "RAG/validation/latest/",
        "RAG/chunked_data/",
        "RAG/index/",
    ]
    
    for folder in folders:
        blob = bucket.blob(folder + ".gitkeep")
        if not blob.exists():
            blob.upload_from_string("")
            print(f"  Created: {folder}")
    
    print()
    print(" GCS folder structure ready!")
    print()
    print(f"Bucket: gs://{bucket_name}")
    print(f"Location: {bucket.location}")
    
except Exception as e:
    print(f"\n Setup failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
EOF

echo ""
echo "=========================================="
echo "GCS Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  cd airflow"
echo "  docker-compose up -d"