# Part 3: GCP Setup Verification - Step by Step Guide

Since your friend has already done most of the setup, we'll verify everything is configured correctly.

## Quick Start

Run the automated verification script:

```bash
cd ~/Documents/group29project/MedScan_ai

# Run verification (will read from airflow/.env automatically)
./scripts/verify_gcp_setup.sh

# The script will:
# 1. Read GCP_PROJECT_ID and GCS_BUCKET_NAME from airflow/.env (around lines 41-42)
# 2. Use those values for all verification checks
```

## Manual Verification Steps

If you prefer to check manually or the script shows issues:

### Step 3.1: Verify GCP Authentication & Project

```bash
# 1. Check if you're authenticated
gcloud auth list

# Should show an active account. If not:
gcloud auth login

# 2. Check current project
gcloud config get-value project

# Should match the value from airflow/.env (check what's in your .env file)
# If not matching, set it to match your .env:
# gcloud config set project <your-project-from-env>

# 3. Verify service account key exists
ls -la ~/gcp-service-account.json

# Should exist. If not, download from GCP Console:
# IAM & Admin → Service Accounts → Select account → Keys → Download JSON
```

### Step 3.2: Verify Environment Variables

```bash
# Check system environment variables
echo "GCP_PROJECT_ID: $GCP_PROJECT_ID"
echo "GCS_BUCKET_NAME: $GCS_BUCKET_NAME"

# If not set, add to ~/.zshrc or ~/.bashrc:
export GCP_PROJECT_ID="medscanai-476203"
export GCS_BUCKET_NAME="medscan-data"

# Check airflow/.env file
cd airflow
grep "GCP_PROJECT_ID" .env
grep "GCS_BUCKET_NAME" .env

# Should show (example - your values may differ):
# GCP_PROJECT_ID=<your-project-id>
# GCS_BUCKET_NAME=<your-bucket-name>
```

### Step 3.3: Verify GCP APIs Are Enabled

```bash
# Check if required APIs are enabled
gcloud services list --enabled --project=medscanai-476203

# Look for these APIs:
# ✓ aiplatform.googleapis.com
# ✓ cloudbuild.googleapis.com
# ✓ storage-component.googleapis.com
# ✓ monitoring.googleapis.com
# ✓ pubsub.googleapis.com
# ✓ secretmanager.googleapis.com

# If any are missing, enable them (use YOUR project ID from airflow/.env):
# PROJECT_ID=$(grep "^GCP_PROJECT_ID=" airflow/.env | cut -d'=' -f2)
# gcloud services enable aiplatform.googleapis.com --project=$PROJECT_ID
# gcloud services enable cloudbuild.googleapis.com --project=$PROJECT_ID
# ... etc
```

### Step 3.4: Verify GCS Bucket & Data

```bash
# Get bucket name from airflow/.env
BUCKET_NAME=$(grep "^GCS_BUCKET_NAME=" airflow/.env | cut -d'=' -f2 | tr -d '"' | tr -d "'")

# Check if bucket exists
gsutil ls -b gs://$BUCKET_NAME

# Should show: gs://$BUCKET_NAME/

# Check if data exists (from DataPipeline)
gsutil ls gs://$BUCKET_NAME/vision/preprocessed/
gsutil ls gs://$BUCKET_NAME/RAG/index/

# Should show:
# vision/preprocessed/tb/
# vision/preprocessed/lung_cancer_ct_scan/
# RAG/index/index.bin
# RAG/index/embeddings.json

# If bucket doesn't exist, create it:
# gsutil mb -l us-central1 gs://$BUCKET_NAME
```

### Step 3.5: Verify Secret Manager

```bash
# Check if SMTP secrets exist
gcloud secrets describe smtp-username --project=medscanai-476203
gcloud secrets describe smtp-password --project=medscanai-476203

# Should show secret details. If not, create them:
# echo "your-email@gmail.com" | gcloud secrets create smtp-username --data-file=- --project=medscanai-476203
# echo "your-app-password" | gcloud secrets create smtp-password --data-file=- --project=medscanai-476203
```

### Step 3.6: Verify Cloud Build Configuration

```bash
# Check Cloud Build triggers (if GitHub is connected)
gcloud builds triggers list --project=medscanai-476203

# Should show triggers like:
# vision-model-training
# rag-model-training

# If not connected yet, that's OK - you can trigger manually first

# Test Cloud Build config file exists
ls -la cloudbuild/vision-training.yaml
ls -la cloudbuild/rag-training.yaml
```

### Step 3.7: Verify Pub/Sub Topics

```bash
# Check Pub/Sub topics
gcloud pubsub topics list --project=medscanai-476203

# Should show:
# model-alerts
# model-retraining-check

# If missing, create them:
# gcloud pubsub topics create model-alerts --project=medscanai-476203
# gcloud pubsub topics create model-retraining-check --project=medscanai-476203
```

### Step 3.8: Verify Python Code Configuration

```bash
# Check if Python files are using environment variables
cd ~/Documents/group29project/MedScan_ai

# This should NOT find hardcoded values in actual code:
grep -r "medscanai-476203" --include="*.py" . | grep -v "getenv\|os.getenv" | grep -v "__pycache__" | grep -v ".env"

# Should only find:
# - Comments
# - Documentation
# - Default fallback values (which is OK)

# Check key files use env vars:
grep "os.getenv.*GCP_PROJECT_ID" ModelDevelopment/common/gcp_utils.py
grep "os.getenv.*GCS_BUCKET_NAME" ModelDevelopment/common/gcp_utils.py
```

## Quick Fixes for Common Issues

### Issue: Project ID Mismatch

```bash
# Update all places:
export GCP_PROJECT_ID="medscanai-476203"
gcloud config set project medscanai-476203

# Update airflow/.env
cd airflow
sed -i.bak 's/GCP_PROJECT_ID=.*/GCP_PROJECT_ID=medscanai-476203/' .env
```

### Issue: Bucket Name Mismatch

```bash
# Update environment variable
export GCS_BUCKET_NAME="medscan-data"

# Update airflow/.env
cd airflow
sed -i.bak 's/GCS_BUCKET_NAME=.*/GCS_BUCKET_NAME=medscan-data/' .env
```

### Issue: Missing APIs

```bash
# Enable all required APIs at once
gcloud services enable \
    aiplatform.googleapis.com \
    cloudbuild.googleapis.com \
    storage-component.googleapis.com \
    monitoring.googleapis.com \
    pubsub.googleapis.com \
    secretmanager.googleapis.com \
    cloudscheduler.googleapis.com \
    --project=medscanai-476203
```

### Issue: Missing Secrets

```bash
# Create SMTP secrets (you'll be prompted for values)
read -s SMTP_USER
echo "$SMTP_USER" | gcloud secrets create smtp-username --data-file=- --project=medscanai-476203

read -s SMTP_PASS
echo "$SMTP_PASS" | gcloud secrets create smtp-password --data-file=- --project=medscanai-476203
```

## Verification Checklist

Use this checklist to verify everything:

- [ ] GCloud authenticated (`gcloud auth list`)
- [ ] Project set correctly (`gcloud config get-value project`)
- [ ] Service account key exists (`~/gcp-service-account.json`)
- [ ] Environment variables set (`GCP_PROJECT_ID`, `GCS_BUCKET_NAME`)
- [ ] `airflow/.env` has correct GCP values
- [ ] All required APIs enabled
- [ ] GCS bucket exists (`gs://medscan-data`)
- [ ] Vision data exists in GCS (if DataPipeline ran)
- [ ] RAG data exists in GCS (if DataPipeline ran)
- [ ] Secret Manager secrets exist
- [ ] Cloud Build config files exist
- [ ] Pub/Sub topics exist (if using monitoring)
- [ ] Python files use environment variables (not hardcoded)

## Next Steps After Verification

Once everything is verified:

1. **Test Cloud Build Manually:**
   ```bash
   gcloud builds submit \
       --config=cloudbuild/vision-training.yaml \
       --substitutions=_DATASET=tb,_EPOCHS=2 \
       --project=medscanai-476203
   ```

2. **Verify GitHub Connection** (if using CI/CD):
   - Go to: https://console.cloud.google.com/cloud-build/triggers
   - Check if triggers are connected to GitHub

3. **Test Monitoring** (optional):
   ```bash
   # Check monitoring dashboard
   open https://console.cloud.google.com/monitoring?project=medscanai-476203
   ```

## Troubleshooting

If the verification script fails:

1. **Read the error message** - it tells you what's missing
2. **Check the manual steps above** for that specific item
3. **Fix one issue at a time** - don't try to fix everything at once
4. **Re-run the verification** after each fix

For help with specific errors, check the error message and look for the corresponding section in this guide.

