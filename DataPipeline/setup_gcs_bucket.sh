#!/bin/bash
set -e

echo "MedScan AI - Unified GCS Bucket Setup"
echo ""

export GOOGLE_APPLICATION_CREDENTIALS=~/gcp-service-account.json

python3 << 'EOF'
from google.cloud import storage

# Single unified bucket
project_id = "medscanai-476203"
bucket_name = "medscan-data"

client = storage.Client(project=project_id)

# Create or get bucket
try:
    bucket = client.get_bucket(bucket_name)
    print(f"✓ Using existing bucket: {bucket_name}")
except:
    bucket = client.create_bucket(bucket_name, location="us-central1")
    print(f"✓ Created bucket: {bucket_name}")

print("")
print("Creating unified folder structure:")
print("")

# Unified folder structure for both pipelines
folders = [
    # RAG Pipeline
    "RAG/config/",
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
]

for folder in folders:
    blob = bucket.blob(folder + ".gitkeep")
    if not blob.exists():
        blob.upload_from_string("")
        print(f"  ✓ {folder}")
    else:
        print(f"  - {folder} (exists)")

print("")
print("✓ GCS Setup Complete!")
print(f"Project: {project_id}")
print(f"Bucket: gs://{bucket_name}")
print(f"  - RAG: gs://{bucket_name}/RAG/")
print(f"  - Vision: gs://{bucket_name}/vision/")
print(f"Total Folders: {len(folders)}")
print("")
print("Next Steps:")
print("  1. Run: python DataPipeline/scripts/RAG/create_urls_file.py")
print("  2. Start Airflow and run DAGs")

EOF

echo ""
echo "Setup complete!"