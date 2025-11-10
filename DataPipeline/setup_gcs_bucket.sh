#!/bin/bash
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
ENV_FILE="${PROJECT_ROOT}/airflow/.env"

echo "MedScan AI - Unified GCS Bucket Setup"
echo ""

if [ -f "${ENV_FILE}" ]; then
  echo "Loading environment from ${ENV_FILE}"
  set -a
  # shellcheck source=/dev/null
  source "${ENV_FILE}"
  set +a
  echo "Environment variables loaded"
  echo ""
else
  echo "No airflow/.env found at ${ENV_FILE}. Using default configuration values."
  echo ""
fi

export GOOGLE_APPLICATION_CREDENTIALS=${GOOGLE_APPLICATION_CREDENTIALS:-$HOME/gcp-service-account.json}
export DATA_PIPELINE_DIR="${SCRIPT_DIR}"

python3 << 'EOF'
import os
import importlib.util
from pathlib import Path

from google.api_core.exceptions import Conflict, NotFound
from google.cloud import storage

script_dir = Path(os.environ["DATA_PIPELINE_DIR"]).resolve()
gcp_config_path = script_dir / "config" / "gcp_config.py"

spec = importlib.util.spec_from_file_location("gcp_config", gcp_config_path)
gcp_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gcp_config)

project_id = os.getenv("GCP_PROJECT_ID", gcp_config.PROJECT_ID)
bucket_name = os.getenv("GCS_BUCKET_NAME", gcp_config.BUCKET_NAME)

client = storage.Client(project=project_id)

# Create or get bucket
try:
    bucket = client.get_bucket(bucket_name)
    print(f"✓ Using existing bucket: {bucket_name}")
except NotFound:
    bucket = client.create_bucket(bucket_name, location="us-central1")
    print(f"✓ Created bucket: {bucket_name}")
except Conflict:
    bucket = client.get_bucket(bucket_name)
    print(f"✓ Bucket already exists and is accessible: {bucket_name}")

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