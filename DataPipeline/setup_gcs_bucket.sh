#!/bin/bash
set -e

echo "=========================================="
echo "GCS Setup for Vision Pipeline"
echo "=========================================="

export GOOGLE_APPLICATION_CREDENTIALS=~/gcp-service-account.json

python << 'EOF'
from google.cloud import storage

# YOUR project and existing bucket
client = storage.Client(project="medscanai-476500")
bucket_name = "medscan-pipeline-medscanai-476500"  # Use existing bucket

# Get existing bucket
bucket = client.get_bucket(bucket_name)
print(f"✓ Using existing bucket: {bucket_name}")
print("")

print("Creating vision pipeline folders:")

# Vision folder structure
folders = [
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
    blob.upload_from_string("")
    print(f"  ✓ {folder}")

print("")
print("="*50)
print("✓ Setup complete!")
print(f"  Project: medscanai-476500")
print(f"  Bucket: gs://{bucket_name}")
print(f"  Vision: gs://{bucket_name}/vision/")
print("="*50)
EOF