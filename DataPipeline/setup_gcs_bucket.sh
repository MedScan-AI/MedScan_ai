#!/bin/bash
set -e

echo "GCS Setup for RAG"

export GOOGLE_APPLICATION_CREDENTIALS=~/gcp-service-account.json

python3 << 'EOF'
from google.cloud import storage

client = storage.Client(project="medscanai-476203")
bucket_name = "medscan-rag-data"

try:
    bucket = client.get_bucket(bucket_name)
    print(f"✓ Bucket exists: {bucket_name}")
except:
    bucket = client.create_bucket(bucket_name, location="us-central1")
    print(f"✓ Created: {bucket_name}")

# Folder structure
folders = [
    "RAG/config/",
    "RAG/raw_data/baseline/",
    "RAG/raw_data/incremental/",
    "RAG/validation/",
    "RAG/validation/reports/",
    "RAG/index/",
]

for folder in folders:
    blob = bucket.blob(folder + ".gitkeep")
    if not blob.exists():
        blob.upload_from_string("")
        print(f"  Created: {folder}")

print("✓ GCS ready!")
EOF

echo ""
echo "Next: Run create_urls_file.py to upload URL list"