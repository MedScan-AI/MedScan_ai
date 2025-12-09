# Vision Inference API - CI/CD Workflow

This document describes the automated deployment workflow for the Vision Inference API.

## 📋 Workflow: `vision-inference-deploy.yaml`

Automatically deploys the Vision Inference API to Google Cloud Run when changes are detected in the inference codebase.

## 🔄 Trigger Conditions

The workflow triggers on:

### 1. Push to main branch
When code is pushed to `main` with changes to:
- `ModelDevelopment/VisionInference/**` (any file in this directory)
- `.github/workflows/vision-inference-deploy.yaml` (the workflow itself)

### 2. Pull Request
When a PR is opened/updated targeting `main` with changes to:
- `ModelDevelopment/VisionInference/**`

**Note:** PR runs will validate but NOT deploy (dry-run only).

### 3. Manual Trigger
Via GitHub UI: Actions → Vision Inference API - Deploy → Run workflow
- Optional: Check "Force deployment" to deploy even without changes

## 🏗️ Deployment Steps

The workflow performs these steps:

### 1. **Environment Setup**
- ✅ Checkout code
- ✅ Authenticate to GCP using service account
- ✅ Configure gcloud CLI

### 2. **Pre-deployment Validation**
- ✅ Check for trained models in GCS bucket
- ✅ Verify Artifact Registry repository exists
- ✅ Create repository if missing
- ✅ Grant necessary IAM permissions to Cloud Build

### 3. **Build and Deploy**
- ✅ Submit Cloud Build using `cloudbuild.yaml`
- ✅ Build Docker image
- ✅ Push to Artifact Registry
- ✅ Deploy to Cloud Run
- ✅ Wait for service to be ready

### 4. **Post-deployment Testing**
- ✅ Get service URL
- ✅ Test health endpoint (`/health`)
- ✅ Test TB prediction endpoint (`/predict/tb`)
- ✅ Test Lung Cancer prediction endpoint (`/predict/lung_cancer`)
- ✅ Check recent logs for errors

### 5. **Reporting**
- ✅ Display deployment summary
- ✅ Create deployment artifact with metadata
- ✅ Upload deployment info (available for 30 days)

## 🔐 Required Secrets

Set these secrets in your GitHub repository:

### `GCP_SA_KEY`
Service account JSON key with permissions:
- `roles/run.admin` - Deploy to Cloud Run
- `roles/artifactregistry.admin` - Push Docker images
- `roles/cloudbuild.builds.editor` - Submit Cloud Builds
- `roles/storage.objectViewer` - Read models from GCS
- `roles/iam.serviceAccountUser` - Act as service accounts

**To create the service account:**

```bash
# Create service account
gcloud iam service-accounts create github-actions-vision \
  --display-name="GitHub Actions - Vision Inference" \
  --project=medscanai-476500

# Grant necessary roles
for role in \
  "roles/run.admin" \
  "roles/artifactregistry.admin" \
  "roles/cloudbuild.builds.editor" \
  "roles/storage.objectViewer" \
  "roles/iam.serviceAccountUser" \
  "roles/iam.securityAdmin"; do
  gcloud projects add-iam-policy-binding medscanai-476500 \
    --member="serviceAccount:github-actions-vision@medscanai-476500.iam.gserviceaccount.com" \
    --role="$role"
done

# Create and download key
gcloud iam service-accounts keys create github-actions-key.json \
  --iam-account=github-actions-vision@medscanai-476500.iam.gserviceaccount.com

# Add to GitHub Secrets
# Go to: Settings → Secrets and variables → Actions → New repository secret
# Name: GCP_SA_KEY
# Value: <paste entire contents of github-actions-key.json>
```

## 📊 Workflow Outputs

### Deployment Summary
After successful deployment, the workflow displays:
- ✅ Service URL
- ✅ Health endpoint
- ✅ API documentation URL
- ✅ Prediction endpoints (TB, Lung Cancer)
- ✅ Build ID
- ✅ Region

### Deployment Artifact
A JSON file with deployment metadata:
```json
{
  "service_name": "vision-inference-api",
  "service_url": "https://vision-inference-api-123456.us-central1.run.app",
  "region": "us-central1",
  "project_id": "medscanai-476500",
  "build_id": "abc123...",
  "deployed_at": "2025-12-08T18:30:00Z",
  "commit_sha": "abc123...",
  "commit_message": "Update model loader",
  "triggered_by": "username"
}
```

**Download from:** Actions → Workflow run → Artifacts

## 🧪 Testing

The workflow automatically tests:

1. **Health Check** (`GET /health`)
   - Expected: 200 OK
   - Verifies: Service is running

2. **TB Endpoint** (`POST /predict/tb`)
   - Expected: 200, 400, or 422 (endpoint accessible)
   - Verifies: TB model endpoint is available

3. **Lung Cancer Endpoint** (`POST /predict/lung_cancer`)
   - Expected: 200, 400, or 422 (endpoint accessible)
   - Verifies: Lung cancer model endpoint is available

## 🐛 Troubleshooting

### Workflow fails with "Permission denied"

**Solution:** Check service account has all required roles (see [Required Secrets](#-required-secrets))

### "No trained models found" warning

**Cause:** Models haven't been uploaded to GCS yet

**Solution:** 
- Run Vision training pipeline first: `.github/workflows/vision-training.yaml`
- Or manually upload models to: `gs://medscan-pipeline-medscanai-476500/vision/trained_models/`

### Health check fails (HTTP 500)

**Cause:** Service started but models failed to load

**Solution:**
1. Check logs: `gcloud run services logs read vision-inference-api`
2. Verify models exist in GCS
3. Check Cloud Run service has `roles/storage.objectViewer` on the GCS bucket

### Build times out

**Cause:** Slow build (Docker layer caching not working)

**Solution:**
- Cloud Build timeout is 1800s (30 min) in `cloudbuild.yaml`
- Check machine type in `cloudbuild.yaml` (currently no machineType = default)
- Consider using `machineType: 'e2-highcpu-4'` for faster builds

## 📈 Performance

**Typical workflow runtime:**
- Pre-deployment checks: ~30s
- Cloud Build: ~8-12 minutes
- Post-deployment tests: ~30s
- **Total: ~10-15 minutes**

**Cold start (first deployment):**
- May take longer as Docker layers are built from scratch
- Subsequent deployments use cached layers

## 🔄 Manual Deployment

If the workflow fails or you need to deploy manually:

```bash
# From repository root
cd ModelDevelopment/VisionInference

# Submit build
gcloud builds submit \
  --config=cloudbuild.yaml \
  --project=medscanai-476500 \
  --region=us-central1
```

## 📝 Workflow Configuration

Edit `.github/workflows/vision-inference-deploy.yaml` to customize:

```yaml
env:
  PROJECT_ID: medscanai-476500        # GCP project
  REGION: us-central1                 # Cloud Run region
  SERVICE_NAME: vision-inference-api  # Service name
  ARTIFACT_REGISTRY_REPO: vision-inference  # Docker repo
```

## 🔗 Related Workflows

- **Vision Training** (`.github/workflows/vision-training.yaml`)
  - Trains models and uploads to GCS
  - Should run before inference deployment

- **RAG Deployment** (`.github/workflows/rag-deploy.yaml`)
  - Similar pattern for RAG service

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Cloud Run CI/CD Guide](https://cloud.google.com/run/docs/continuous-deployment)
- [Cloud Build Configuration](https://cloud.google.com/build/docs/build-config-file-schema)

## 💡 Best Practices

1. **Always test locally first** before pushing
   ```bash
   docker-compose up --build
   curl http://localhost:5000/health
   ```

2. **Use PR reviews** for inference code changes
   - Workflow runs on PRs but doesn't deploy
   - Review changes before merging to main

3. **Monitor deployments** in Cloud Console
   - Cloud Build: https://console.cloud.google.com/cloud-build
   - Cloud Run: https://console.cloud.google.com/run

4. **Check logs after deployment**
   ```bash
   gcloud run services logs read vision-inference-api \
     --region=us-central1 \
     --limit=50
   ```

5. **Use workflow artifacts** to track deployments
   - Download deployment JSON from Actions tab
   - Keep for audit trail
