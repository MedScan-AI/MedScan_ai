# Vision Inference API - CI/CD Documentation

This document describes the automated deployment pipeline for the Vision Inference API.

## 📦 Files Created

### GitHub Actions Workflows
- **`.github/workflows/vision-inference-terraform-setup.yaml`**
  - Infrastructure provisioning workflow (Terraform)
  - One-time setup or manual trigger
  - Creates Cloud Run, Artifact Registry, IAM permissions
  
- **`.github/workflows/vision-inference-deploy.yaml`**
  - Application deployment workflow
  - Triggers on changes to `ModelDevelopment/VisionInference/`
  - Automatically deploys to Cloud Run

### Documentation
- **`.github/workflows/README-vision-inference.md`**
  - Complete workflow documentation
  - Setup instructions
  - Troubleshooting guide

### Setup Script
- **`.github/workflows/setup-vision-inference-cicd.sh`**
  - Automated setup script
  - Creates service account
  - Grants necessary permissions

## 🚀 Quick Start

### 0. (Optional) Setup Infrastructure with Terraform

If infrastructure doesn't exist yet, run the Terraform workflow:

**Via GitHub Actions UI:**
1. Go to: **Actions → Vision Inference - Terraform Setup**
2. Click: **Run workflow**
3. Select:
   - Action: `plan` (to preview changes)
   - Then run again with: `apply` + auto_approve=true

**Or push changes to:**
```bash
git add deploymentVisionInference/terraform/
git commit -m "Update Terraform config"
git push origin main
# This will run `terraform plan` only
```

**What it creates:**
- ✅ Artifact Registry repository
- ✅ Cloud Run service (placeholder, no container yet)
- ✅ IAM permissions for Cloud Build
- ✅ Enabled APIs

### 1. Run Setup Script

```bash
cd .github/workflows
chmod +x setup-vision-inference-cicd.sh
./setup-vision-inference-cicd.sh
```

This will:
- ✅ Create a service account: `github-actions-vision@medscanai-476500.iam.gserviceaccount.com`
- ✅ Grant necessary IAM roles
- ✅ Generate service account key JSON

### 2. Add Secret to GitHub

1. Copy the service account key:
   ```bash
   cat github-actions-vision-key.json
   ```

2. Go to GitHub: **Settings → Secrets and variables → Actions → New repository secret**

3. Add secret:
   - **Name:** `GCP_SA_KEY`
   - **Value:** (paste entire JSON contents)

4. Delete the local key file:
   ```bash
   rm github-actions-vision-key.json
   ```

### 3. Test the Workflow

#### Option A: Push to main
```bash
# Make a small change to the inference code
echo "# Test" >> ModelDevelopment/VisionInference/README.md
git add ModelDevelopment/VisionInference/README.md
git commit -m "Test CI/CD deployment"
git push origin main
```

#### Option B: Manual trigger
1. Go to: **Actions → Vision Inference API - Deploy**
2. Click: **Run workflow**
3. Select branch: `main`
4. Click: **Run workflow**

### 4. Monitor Deployment

Watch the progress at:
```
https://github.com/YOUR_ORG/YOUR_REPO/actions
```

## 🔄 How It Works

### Automatic Deployment Flow

```
Code Change → GitHub Push → Workflow Triggers → Build & Deploy → Test → Complete
     ↓              ↓              ↓                   ↓           ↓         ↓
  Engineer     Git commit    Path filter      Cloud Build    Endpoints   Logs
                             matches           Docker build    tested    checked
                                              Push to AR
                                              Deploy to CR
```

### Trigger Conditions

The workflow automatically runs when:

1. **Push to `main` branch** with changes to:
   - Any file in `ModelDevelopment/VisionInference/`
   - The workflow file itself

2. **Pull Request** to `main` with changes to:
   - Any file in `ModelDevelopment/VisionInference/`
   - (Validates but doesn't deploy)

3. **Manual trigger** via GitHub Actions UI

### What Gets Deployed

- ✅ FastAPI inference application
- ✅ Model loader (GCS integration)
- ✅ GradCAM visualization
- ✅ Health check endpoint
- ✅ TB prediction endpoint
- ✅ Lung Cancer prediction endpoint

## 📊 Workflow Steps

### Phase 1: Pre-deployment (30s)
1. Checkout code
2. Authenticate to GCP
3. Check for trained models in GCS
4. Verify Artifact Registry repository
5. Grant Cloud Build permissions

### Phase 2: Build & Deploy (8-12 min)
1. Submit Cloud Build
2. Build Docker image
3. Push to Artifact Registry
4. Deploy to Cloud Run
5. Wait for service ready

### Phase 3: Testing (30s)
1. Get service URL
2. Test health endpoint
3. Test TB prediction endpoint
4. Test Lung Cancer prediction endpoint
5. Check logs for errors

### Phase 4: Reporting
1. Display deployment summary
2. Save deployment metadata
3. Upload artifact (JSON)

## 🎯 Deployment Targets

### Current Configuration

| Parameter | Value |
|-----------|-------|
| **Project** | `medscanai-476500` |
| **Region** | `us-central1` |
| **Service** | `vision-inference-api` |
| **Memory** | `2Gi` |
| **CPU** | `1` |
| **Min Instances** | `0` (scale to zero) |
| **Max Instances** | `3` |
| **Timeout** | `300s` |

### Service Endpoints

After deployment:
- **Health:** `https://vision-inference-api-246542889931.us-central1.run.app/health`
- **Docs:** `https://vision-inference-api-246542889931.us-central1.run.app/docs`
- **TB:** `https://vision-inference-api-246542889931.us-central1.run.app/predict/tb`
- **Lung Cancer:** `https://vision-inference-api-246542889931.us-central1.run.app/predict/lung_cancer`

## 🔐 Security

### Service Account Permissions

The GitHub Actions service account has:
- `roles/run.admin` - Deploy to Cloud Run
- `roles/artifactregistry.admin` - Push Docker images
- `roles/cloudbuild.builds.editor` - Submit Cloud Builds
- `roles/storage.objectViewer` - Read models from GCS
- `roles/iam.serviceAccountUser` - Act as service accounts

### Best Practices

1. ✅ Service account key stored as GitHub Secret
2. ✅ Least privilege access (only necessary roles)
3. ✅ Automatic key rotation (recommended every 90 days)
4. ✅ No keys committed to repository
5. ✅ Audit logs enabled in GCP

## 🐛 Troubleshooting

### "Permission denied" errors

**Check:** Service account has all required roles
```bash
gcloud projects get-iam-policy medscanai-476500 \
  --flatten="bindings[].members" \
  --filter="bindings.members:github-actions-vision@*"
```

### Build times out

**Solution:** Build takes >30 min (rare)
- Check Cloud Build logs: https://console.cloud.google.com/cloud-build
- Increase timeout in `cloudbuild.yaml`

### Health check fails

**Causes:**
- Models failed to load from GCS
- Service account lacks GCS read permissions
- Models don't exist at expected GCS path

**Check logs:**
```bash
gcloud run services logs read vision-inference-api \
  --region=us-central1 \
  --project=medscanai-476500 \
  --limit=50
```

### Workflow doesn't trigger

**Checklist:**
- ✅ Changes pushed to `main` branch
- ✅ Changes in `ModelDevelopment/VisionInference/` directory
- ✅ Workflow file exists: `.github/workflows/vision-inference-deploy.yaml`
- ✅ Repository has GitHub Actions enabled

## 📈 Metrics & Monitoring

### Deployment Frequency
Track at: **Insights → Actions → Workflows**

### Success Rate
View at: **Actions → Vision Inference API - Deploy → Filter: Failed**

### Build Times
Average: 10-15 minutes
- Pre-checks: ~30s
- Build: ~8-12 min
- Tests: ~30s

### Cost Estimate
Per deployment:
- Cloud Build: ~$0.01 - $0.02
- Cloud Run: ~$0.00 (scale to zero when idle)
- Artifact Registry: ~$0.00 (storage negligible)

**Total monthly cost (10 deployments):** ~$0.10 - $0.20

## 🔄 Rollback Strategy

If deployment fails or introduces bugs:

### Option 1: Revert commit
```bash
git revert HEAD
git push origin main
# Workflow automatically deploys previous version
```

### Option 2: Manual rollback
```bash
# List revisions
gcloud run revisions list \
  --service=vision-inference-api \
  --region=us-central1

# Rollback to previous revision
gcloud run services update-traffic vision-inference-api \
  --region=us-central1 \
  --to-revisions=vision-inference-api-00001-abc=100
```

### Option 3: Redeploy specific commit
```bash
# From GitHub Actions UI
# Actions → Vision Inference API - Deploy → Run workflow
# Select: specific commit SHA
```

## 📚 Related Documentation

- [Workflow README](.github/workflows/README-vision-inference.md) - Detailed workflow docs
- [Terraform IaC](terraform/README.md) - Infrastructure as Code
- [API Documentation](README.md) - Inference API usage
- [Training Pipeline](.github/workflows/vision-training.yaml) - Model training CI/CD

## 💡 Tips & Best Practices

### Development Workflow

1. **Local testing first**
   ```bash
   cd ModelDevelopment/VisionInference
   docker-compose up --build
   curl http://localhost:5000/health
   ```

2. **Create feature branch**
   ```bash
   git checkout -b feature/improve-inference
   ```

3. **Make changes and test**
   ```bash
   # Edit code
   docker-compose up --build
   # Test locally
   ```

4. **Create Pull Request**
   ```bash
   git push origin feature/improve-inference
   # Open PR on GitHub
   # Workflow runs validation (no deployment)
   ```

5. **Merge to main**
   ```bash
   # After PR approval
   # Workflow automatically deploys to production
   ```

### Monitoring

- **Cloud Run Metrics:** https://console.cloud.google.com/run/detail/us-central1/vision-inference-api/metrics
- **Cloud Build History:** https://console.cloud.google.com/cloud-build/builds
- **GitHub Actions:** https://github.com/YOUR_ORG/YOUR_REPO/actions

### Cost Optimization

1. ✅ Scale to zero (min_instances=0)
2. ✅ Use smaller machine (1 CPU, 2GB RAM)
3. ✅ Cache Docker layers in Cloud Build
4. ✅ Only deploy on actual code changes (path filters)

## 🎓 Learning Resources

- [GitHub Actions Best Practices](https://docs.github.com/en/actions/security-guides/security-hardening-for-github-actions)
- [Cloud Run CI/CD](https://cloud.google.com/run/docs/continuous-deployment-with-cloud-build)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Terraform GCP Provider](https://registry.terraform.io/providers/hashicorp/google/latest/docs)

## 📋 Workflow Comparison

### When to Use Which Workflow?

| Workflow | Purpose | When to Run | Frequency |
|----------|---------|-------------|-----------|
| **vision-inference-terraform-setup.yaml** | Infrastructure provisioning | First-time setup, infrastructure changes | Once / Rarely |
| **vision-inference-deploy.yaml** | Application deployment | Code changes, model updates | Every push to main |

### Typical Setup Sequence

```
1. Run Terraform Setup (once)
   ↓
   Creates: Cloud Run, Artifact Registry, IAM

2. Run Application Deploy (automatic)
   ↓
   Builds: Docker image, deploys to Cloud Run

3. Code changes (automatic)
   ↓
   Triggers: Application Deploy workflow
```

### Manual Workflow Triggers

**Terraform Setup:**
- Actions → Vision Inference - Terraform Setup → Run workflow
- Choose: `plan` (preview) or `apply` (create) or `destroy` (delete)

**Application Deploy:**
- Actions → Vision Inference API - Deploy → Run workflow
- Force deployment checkbox available
