# Testing Guide for MedScan AI Components

This guide explains how to test both **Vision** and **RAG** components, including their deployment workflows and monitoring infrastructure.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Testing RAG Component](#testing-rag-component)
3. [Testing Vision Component](#testing-vision-component)
4. [Testing Monitoring Infrastructure](#testing-monitoring-infrastructure)
5. [Verification Checklists](#verification-checklists)
6. [Testing Scenarios](#testing-scenarios)
7. [Quick Test Commands](#quick-test-commands)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before testing, ensure you have:

1. ✅ **GitHub Secrets configured:**
   - `GCP_SA_KEY`: Service account JSON with required permissions
   - `SMTP_USER` and `SMTP_PASSWORD` (optional, for notifications)

2. ✅ **GCP Resources ready:**
   - **RAG**: Model and index files in GCS: `gs://medscan-pipeline-medscanai-476500/RAG/`
   - **Vision**: Trained models in GCS: `gs://medscan-pipeline-medscanai-476500/vision/trained_models/`
   - **RAG**: HuggingFace token in Secret Manager: `huggingface-token`
   - Required GCP APIs enabled (Cloud Run, Cloud Build, Monitoring, Logging)

3. ✅ **Permissions:**
   - Service account has `roles/run.admin`
   - Service account has `roles/cloudbuild.builds.editor`
   - Service account has monitoring roles (workflows will grant these)

---

## Testing RAG Component

### RAG Complete Setup Workflow

**Workflow**: `rag-complete-setup.yaml`  
**Purpose**: One-stop deployment + monitoring setup for RAG service

#### Method 1: GitHub Actions UI (Recommended)

**Step 1: Navigate to Actions**
1. Go to your GitHub repository
2. Click on **"Actions"** tab
3. Find **"RAG Complete Setup - Cloud Run + Monitoring"** in the workflow list

**Step 2: Run the Workflow**
1. Click on **"RAG Complete Setup - Cloud Run + Monitoring"**
2. Click **"Run workflow"** button (top right)
3. Configure inputs:
   - **monitoring_email**: Your email (e.g., `your-email@example.com`)
   - **enable_monitoring**: `true` (default)
   - **auto_approve_terraform**: `true` (for testing, set to false in production)
4. Click **"Run workflow"**

**Step 3: Monitor Execution**
1. Watch the workflow run in real-time
2. Check each job:
   - `deploy-cloud-run`: Should complete successfully
   - `setup-monitoring`: Should complete successfully (if enabled)
   - `summary`: Should show deployment summary

**Step 4: Verify Results**

After workflow completes, verify:

1. **Cloud Run Service:**
   ```bash
   gcloud run services describe rag-service \
     --region=us-central1 \
     --project=medscanai-476500 \
     --format="value(status.url)"
   ```

2. **Service Health:**
   ```bash
   SERVICE_URL=$(gcloud run services describe rag-service \
     --region=us-central1 \
     --project=medscanai-476500 \
     --format="value(status.url)")
   
   curl $SERVICE_URL/health
   curl $SERVICE_URL/config
   ```

3. **Monitoring Dashboard:**
   - Go to: https://console.cloud.google.com/monitoring/dashboards?project=medscanai-476500
   - Look for: "RAG Service - Monitoring Dashboard"

4. **Alert Policies:**
   - Go to: https://console.cloud.google.com/monitoring/alerting?project=medscanai-476500
   - Should see 11 alert policies for RAG service

#### Method 2: Push to Main Branch (Automatic Trigger)

**Step 1: Make a Test Change**
```bash
# Make a small change to trigger the workflow
touch deployment/test.txt
git add deployment/test.txt
git commit -m "test: trigger RAG complete setup"
git push origin main
```

**Step 2: Monitor in Actions Tab**
- Workflow will automatically trigger
- Monitor execution in GitHub Actions UI

**Note:** For push events, Terraform will auto-approve by default.

#### Method 3: Test Individual Components

**Test Cloud Run Deployment Only:**
- Use the existing `rag-deploy.yaml` workflow
- Go to Actions → "RAG Deployment"
- Click "Run workflow"
- This tests just the Cloud Run deployment part

**Test Monitoring Infrastructure Only:**

**Option A: Run Terraform Manually**
```bash
cd deploymentRAG/terraform

# Create terraform.tfvars
cat > terraform.tfvars <<EOF
project_id = "medscanai-476500"
region = "us-central1"
service_name = "rag-service"
enable_monitoring = true
monitoring_email = "your-email@example.com"
create_notification_channel = true
enable_apis = false
EOF

# Initialize and apply
terraform init
terraform plan
terraform apply
```

**Option B: Use GitHub Actions with monitoring only**
- Run `rag-complete-setup.yaml` with `enable_monitoring=true`
- The Cloud Run deployment will run first (required dependency)

### RAG Monitoring Workflow

**Workflow**: `rag-monitoring.yaml`  
**Purpose**: Runs monitoring checks (not infrastructure setup)

**How to Test:**
1. Go to Actions → "RAG Monitoring"
2. Click "Run workflow"
3. Optionally enable `trigger_retrain` to test retraining triggers
4. Workflow runs monitoring script and checks metrics

**Note:** This workflow runs automatically every 6 hours via cron schedule.

---

## Testing Vision Component

### Vision Inference Deployment Workflow

**Workflow**: `vision-inference-deploy.yaml`  
**Purpose**: Deploys Vision Inference API to Cloud Run

#### Method 1: GitHub Actions UI

**Step 1: Navigate to Actions**
1. Go to your GitHub repository
2. Click on **"Actions"** tab
3. Find **"Vision Inference API - Deploy"** in the workflow list

**Step 2: Run the Workflow**
1. Click on **"Vision Inference API - Deploy"**
2. Click **"Run workflow"** button (top right)
3. Select branch: `main`
4. Optionally check "Force deployment" to deploy even without changes
5. Click **"Run workflow"**

**Step 3: Monitor Execution**
1. Watch the workflow run in real-time
2. Check each phase:
   - **Pre-deployment**: Model checks, Artifact Registry verification
   - **Build & Deploy**: Docker build, Cloud Run deployment
   - **Testing**: Health check, TB endpoint, Lung Cancer endpoint

**Step 4: Verify Results**

After workflow completes, verify:

1. **Cloud Run Service:**
   ```bash
   gcloud run services describe vision-inference-api \
     --region=us-central1 \
     --project=medscanai-476500 \
     --format="value(status.url)"
   ```

2. **Service Health:**
   ```bash
   SERVICE_URL=$(gcloud run services describe vision-inference-api \
     --region=us-central1 \
     --project=medscanai-476500 \
     --format="value(status.url)")
   
   curl $SERVICE_URL/health
   ```

3. **Test Endpoints:**
   ```bash
   # Test TB endpoint (will return 422 without file, but endpoint is accessible)
   curl -X POST "$SERVICE_URL/predict/tb" \
     -H "Content-Type: multipart/form-data"
   
   # Test Lung Cancer endpoint
   curl -X POST "$SERVICE_URL/predict/lung_cancer" \
     -H "Content-Type: multipart/form-data"
   ```

#### Method 2: Push to Main Branch (Automatic Trigger)

The workflow automatically triggers on push to `main` when changes are made to:
- `ModelDevelopment/VisionInference/**`
- `.github/workflows/vision-inference-deploy.yaml`

### Vision Monitoring Infrastructure Setup

**Workflow**: `vision-inference-terraform-setup.yaml`  
**Purpose**: Sets up monitoring infrastructure via Terraform (manual trigger only)

**How to Test:**

1. **Navigate to Actions**
   - Go to Actions → "Vision Inference - Terraform Setup"

2. **Run Workflow**
   - Click "Run workflow"
   - Select action: `plan` (to preview changes) or `apply` (to create resources)
   - Set `auto_approve`: `true` (for testing)
   - Click "Run workflow"

3. **Verify Monitoring Resources**
   ```bash
   # Check alert policies
   gcloud alpha monitoring policies list --project=medscanai-476500 | grep "Vision Inference"
   
   # Check custom metrics
   gcloud logging metrics list --project=medscanai-476500 | grep "low_confidence"
   
   # Check dashboard
   gcloud monitoring dashboards list --project=medscanai-476500 | grep "Vision Inference"
   ```

**Expected Resources:**
- 6 alert policies (high error rate, high latency, service unavailable, high CPU, high memory, low confidence streak)
- 1 custom metric (`low_confidence_predictions`)
- 1 monitoring dashboard (6 widgets)

### Vision Retraining Threshold Monitor

**Workflow**: `vision-inference-retrain-decoy.yaml`  
**Purpose**: Monitors low-confidence predictions and triggers alerts

**How to Test:**
1. Go to Actions → "Vision Inference Retraining Threshold Monitor"
2. Click "Run workflow"
3. Optionally configure:
   - `threshold`: Low confidence threshold count (default: 50)
   - `hours_lookback`: Hours to look back (default: 24)
4. Workflow checks logs for low-confidence predictions and sends alerts if threshold exceeded

**Note:** This workflow runs automatically every 24 hours via cron schedule.

---

## Testing Monitoring Infrastructure

### RAG Monitoring Infrastructure

**Expected Resources:**
- **7 Custom Log-Based Metrics**:
  1. `rag_composite_score` - Composite quality score (0-1)
  2. `rag_hallucination_score` - Hallucination score (0-1)
  3. `rag_retrieval_score` - Average retrieval score (0-1)
  4. `rag_low_composite_score` - Count of low-quality predictions
  5. `rag_tokens_used` - Total tokens consumed
  6. `rag_docs_retrieved` - Number of documents retrieved
  7. `rag_retraining_triggered` - Retraining event count

- **11 Alert Policies**:
  - 5 Production alerts (error rate, latency, service unavailable, CPU, memory)
  - 5 Quality alerts (composite score, hallucination, retrieval, low quality spike)
  - 1 Retraining alert

- **1 Monitoring Dashboard**: 11 widgets showing production and quality metrics

**Test Commands:**
```bash
# List custom metrics
gcloud logging metrics list --project=medscanai-476500 | grep rag_

# List alert policies
gcloud alpha monitoring policies list --project=medscanai-476500 | grep "RAG Service"

# List dashboards
gcloud monitoring dashboards list --project=medscanai-476500 | grep "RAG Service"

# View dashboard in browser
open "https://console.cloud.google.com/monitoring/dashboards?project=medscanai-476500"
```

### Vision Monitoring Infrastructure

**Expected Resources:**
- **1 Custom Log-Based Metric**:
  - `low_confidence_predictions` - Counts predictions with confidence below threshold

- **6 Alert Policies**:
  1. High Error Rate (>5% for 5 min)
  2. High Latency (P95 >5s for 5 min)
  3. Service Unavailable (no requests in 5h)
  4. High CPU (>80% for 5 min)
  5. High Memory (>85% for 5 min)
  6. Low Confidence Streak (>=3 in 30s)

- **1 Monitoring Dashboard**: 6 widgets showing production metrics

**Test Commands:**
```bash
# List custom metrics
gcloud logging metrics list --project=medscanai-476500 | grep "low_confidence"

# List alert policies
gcloud alpha monitoring policies list --project=medscanai-476500 | grep "Vision Inference"

# List dashboards
gcloud monitoring dashboards list --project=medscanai-476500 | grep "Vision Inference"

# View dashboard in browser
open "https://console.cloud.google.com/monitoring/dashboards?project=medscanai-476500"
```

---

## Verification Checklists

### ✅ RAG Cloud Run Service

- [ ] Service is deployed: `gcloud run services list --region=us-central1`
- [ ] Service URL is accessible
- [ ] Health endpoint returns `{"ready": true}`
- [ ] Config endpoint returns model configuration
- [ ] Prediction endpoint works (test with sample query)

### ✅ RAG Monitoring Infrastructure

- [ ] Dashboard exists in GCP Console: "RAG Service - Monitoring Dashboard"
- [ ] 7 custom metrics created (check Monitoring → Metrics)
- [ ] 11 alert policies created (check Monitoring → Alerting)
- [ ] Email notification channel created (if email provided)

### ✅ Vision Cloud Run Service

- [ ] Service is deployed: `gcloud run services list --region=us-central1`
- [ ] Service URL is accessible
- [ ] Health endpoint returns 200 OK
- [ ] TB prediction endpoint is accessible
- [ ] Lung Cancer prediction endpoint is accessible

### ✅ Vision Monitoring Infrastructure

- [ ] Dashboard exists in GCP Console: "Vision Inference API - Monitoring Dashboard"
- [ ] 1 custom metric created (`low_confidence_predictions`)
- [ ] 6 alert policies created (check Monitoring → Alerting)
- [ ] Email notification channel created (if email provided)

### ✅ Permissions

- [ ] Services are publicly accessible (or IAM configured correctly)
- [ ] Service account has required permissions
- [ ] Monitoring permissions granted

---

## Testing Scenarios

### Scenario 1: RAG First-Time Setup

**Goal:** Test complete setup from scratch

**Steps:**
1. Ensure Cloud Run service doesn't exist (or delete it first)
2. Ensure monitoring infrastructure doesn't exist
3. Run `rag-complete-setup.yaml` with:
   - `monitoring_email`: Your email
   - `enable_monitoring`: `true`
   - `auto_approve_terraform`: `true`
4. Verify both Cloud Run and monitoring are set up

**Expected Result:**
- Cloud Run service deployed and accessible
- Monitoring dashboard created
- 11 alert policies created
- 7 custom metrics created
- Email notifications configured

---

### Scenario 2: Vision First-Time Setup

**Goal:** Test complete Vision setup from scratch

**Steps:**
1. Ensure Cloud Run service doesn't exist (or delete it first)
2. Run `vision-inference-deploy.yaml` to deploy service
3. Run `vision-inference-terraform-setup.yaml` with action `apply` to set up monitoring
4. Verify both Cloud Run and monitoring are set up

**Expected Result:**
- Cloud Run service deployed and accessible
- Monitoring dashboard created
- 6 alert policies created
- 1 custom metric created
- Email notifications configured

---

### Scenario 3: RAG Service Update Only

**Goal:** Test updating existing service

**Steps:**
1. Ensure Cloud Run service already exists
2. Run `rag-complete-setup.yaml` with:
   - `enable_monitoring`: `false` (skip monitoring)
   - `auto_approve_terraform`: `true`
3. Verify service is updated

**Expected Result:**
- Cloud Run service updated with new image
- Monitoring infrastructure unchanged

---

### Scenario 4: Monitoring Setup Only

**Goal:** Test setting up monitoring for existing service

**Steps:**
1. Ensure Cloud Run service already exists
2. Run `rag-complete-setup.yaml` with:
   - `monitoring_email`: Your email
   - `enable_monitoring`: `true`
   - `auto_approve_terraform`: `true`
3. Verify monitoring is set up

**Expected Result:**
- Cloud Run service unchanged (already exists)
- Monitoring dashboard created
- Alert policies created

---

### Scenario 5: Error Handling

**Goal:** Test error handling

**Test Cases:**

1. **Missing GCP credentials:**
   - Remove or invalidate `GCP_SA_KEY` secret
   - Run workflow → Should fail with auth error

2. **Missing model data:**
   - **RAG**: Remove RAG model/index from GCS
   - **Vision**: Remove Vision models from GCS
   - Run workflow → Should fail when service tries to load model

3. **Terraform without auto-approve:**
   - Run with `auto_approve_terraform`: `false`
   - Workflow should stop at Terraform apply step
   - Should show instructions for manual approval

---

## Quick Test Commands

### Test RAG Service Endpoints

```bash
# Get service URL
SERVICE_URL=$(gcloud run services describe rag-service \
  --region=us-central1 \
  --project=medscanai-476500 \
  --format="value(status.url)")

# Test health
curl $SERVICE_URL/health | jq

# Test config
curl $SERVICE_URL/config | jq

# Test prediction
curl -X POST $SERVICE_URL/predict \
  -H "Content-Type: application/json" \
  -d '{"instances":[{"query":"What is tuberculosis?"}]}' | jq
```

### Test Vision Service Endpoints

```bash
# Get service URL
SERVICE_URL=$(gcloud run services describe vision-inference-api \
  --region=us-central1 \
  --project=medscanai-476500 \
  --format="value(status.url)")

# Test health
curl $SERVICE_URL/health

# Test TB endpoint (will return 422 without file, but endpoint is accessible)
curl -X POST "$SERVICE_URL/predict/tb" \
  -H "Content-Type: multipart/form-data"

# Test Lung Cancer endpoint
curl -X POST "$SERVICE_URL/predict/lung_cancer" \
  -H "Content-Type: multipart/form-data"
```

### Check RAG Monitoring Resources

```bash
# List custom metrics
gcloud logging metrics list --project=medscanai-476500 | grep rag_

# List alert policies
gcloud alpha monitoring policies list --project=medscanai-476500 | grep "RAG Service"

# List dashboards
gcloud monitoring dashboards list --project=medscanai-476500 | grep "RAG Service"
```

### Check Vision Monitoring Resources

```bash
# List custom metrics
gcloud logging metrics list --project=medscanai-476500 | grep "low_confidence"

# List alert policies
gcloud alpha monitoring policies list --project=medscanai-476500 | grep "Vision Inference"

# List dashboards
gcloud monitoring dashboards list --project=medscanai-476500 | grep "Vision Inference"
```

### Check Terraform State

```bash
# RAG Terraform
cd deploymentRAG/terraform
terraform state list
terraform output

# Vision Terraform
cd deploymentVisionInference/terraform
terraform state list
terraform output
```

### Use Test Scripts

**RAG Test Script:**
```bash
./scripts/test-rag-deployment.sh
```

This script automatically tests:
- Service existence
- Health endpoint
- Config endpoint
- Prediction endpoint
- Monitoring resources (metrics, alerts, dashboard)

---

## Troubleshooting

### Workflow Fails at Cloud Run Deployment

**Symptoms:**
- Job `deploy-cloud-run` fails
- Cloud Build fails

**Solutions:**
1. Check Cloud Build logs in GCP Console
2. **RAG**: Verify RAG model/index exist in GCS
3. **Vision**: Verify Vision models exist in GCS
4. **RAG**: Check HuggingFace token in Secret Manager
5. Verify service account has `roles/run.admin`

**Debug:**
```bash
# Check Cloud Build logs
gcloud builds list --project=medscanai-476500 --limit=5

# Check service account permissions
gcloud projects get-iam-policy medscanai-476500 \
  --flatten="bindings[].members" \
  --filter="bindings.members:*@*" \
  --format="table(bindings.role)"
```

---

### Workflow Fails at Terraform Apply

**Symptoms:**
- Job `setup-monitoring` fails
- Terraform apply fails

**Solutions:**
1. Check Terraform logs in workflow output
2. Verify service account has monitoring permissions
3. Check if resources already exist (may need to import)
4. Verify APIs are enabled

**Debug:**
```bash
cd deploymentRAG/terraform  # or deploymentVisionInference/terraform
terraform init
terraform plan -detailed-exitcode
terraform validate
```

---

### Service Deployed but Not Accessible

**Symptoms:**
- Workflow succeeds
- Service URL returns 403 or 404

**Solutions:**
1. Check IAM permissions (workflow sets public access)
2. Verify service is actually running
3. Check service logs

**Debug:**
```bash
# Check service status
gcloud run services describe rag-service \
  --region=us-central1 \
  --project=medscanai-476500

# Check IAM policy
gcloud run services get-iam-policy rag-service \
  --region=us-central1 \
  --project=medscanai-476500

# Check service logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=rag-service" \
  --project=medscanai-476500 \
  --limit=50
```

---

### Monitoring Dashboard Not Showing Data

**Symptoms:**
- Dashboard exists but shows no data
- Metrics not appearing

**Solutions:**
1. Wait 5-10 minutes for metrics to start appearing
2. Verify service is logging predictions
3. Check log format matches expected structure
4. Verify custom metrics are created

**Debug:**
```bash
# Check if metrics exist
gcloud logging metrics list --project=medscanai-476500 | grep rag_  # or low_confidence

# Check recent logs (RAG)
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=rag-service AND jsonPayload.prediction_result:*" \
  --project=medscanai-476500 \
  --limit=10 \
  --format=json

# Check recent logs (Vision)
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=vision-inference-api AND textPayload:\"[low_confidence]\"" \
  --project=medscanai-476500 \
  --limit=10

# Test prediction to generate logs
SERVICE_URL=$(gcloud run services describe rag-service \
  --region=us-central1 \
  --project=medscanai-476500 \
  --format="value(status.url)")
curl -X POST $SERVICE_URL/predict \
  -H "Content-Type: application/json" \
  -d '{"instances":[{"query":"test query"}]}'
```

---

### Alert Policies Not Triggering

**Symptoms:**
- Alert policies exist but never trigger
- No email notifications received

**Solutions:**
1. Verify email notification channel is verified in GCP Console
2. Check alert policy is enabled
3. Verify thresholds are being exceeded
4. Check alert policy conditions match log format

**Debug:**
```bash
# Check alert policy status
gcloud alpha monitoring policies list --project=medscanai-476500 \
  --format="table(displayName,enabled,conditions.displayName)"

# Check notification channels
gcloud alpha monitoring channels list --project=medscanai-476500

# Test notification channel
# Go to GCP Console → Monitoring → Alerting → Notification Channels
# Click on channel → "Send Test Notification"
```

---

## Best Practices for Testing

1. **Test in stages:**
   - First test Cloud Run deployment only
   - Then test monitoring setup separately
   - Finally test complete workflow

2. **Use test email:**
   - Use a test email address for notifications
   - Verify alerts are received
   - Check spam folder if emails don't arrive

3. **Monitor costs:**
   - Cloud Run with GPU can be expensive
   - Set up billing alerts
   - Clean up test resources when done

4. **Check logs:**
   - Always check workflow logs in GitHub Actions
   - Check Cloud Build logs in GCP Console
   - Check Cloud Run logs for service issues

5. **Verify incrementally:**
   - Don't wait for entire workflow to complete
   - Check each job as it completes
   - Fix issues before proceeding

6. **Test both components:**
   - Test RAG and Vision separately
   - Verify monitoring works for both
   - Test cross-component interactions if applicable

---

## Cleanup After Testing

If you want to clean up test resources:

### RAG Cleanup

```bash
# Delete Cloud Run service
gcloud run services delete rag-service \
  --region=us-central1 \
  --project=medscanai-476500

# Delete monitoring resources (via Terraform)
cd deploymentRAG/terraform
terraform destroy
```

### Vision Cleanup

```bash
# Delete Cloud Run service
gcloud run services delete vision-inference-api \
  --region=us-central1 \
  --project=medscanai-476500

# Delete monitoring resources (via Terraform)
cd deploymentVisionInference/terraform
terraform destroy
```

**Note:** Be careful with cleanup - make sure you're not deleting production resources!

---

## Next Steps After Successful Test

1. ✅ Document any issues found
2. ✅ Update workflows if needed
3. ✅ Set up scheduled monitoring:
   - RAG: `rag-monitoring.yaml` (runs every 6 hours)
   - Vision: `vision-inference-retrain-decoy.yaml` (runs every 24 hours)
4. ✅ Configure production alerts
5. ✅ Test retraining workflows if needed
6. ✅ Set up production email notifications

---

## Support

If you encounter issues:

1. Check workflow logs in GitHub Actions
2. Check Cloud Build logs in GCP Console
3. Check Cloud Run logs
4. Review Terraform state and outputs
5. Check this guide's troubleshooting section
6. Review component-specific documentation:
   - RAG: `deploymentRAG/terraform/README.md`
   - Vision: `deploymentVisionInference/terraform/README-monitoring-setup.md`

---

## Quick Reference

### RAG Workflows
- **Complete Setup**: `rag-complete-setup.yaml` - Deploy + monitoring
- **Deployment Only**: `rag-deploy.yaml` - Cloud Run only
- **Monitoring Checks**: `rag-monitoring.yaml` - Runs monitoring script

### Vision Workflows
- **Deployment**: `vision-inference-deploy.yaml` - Deploy to Cloud Run
- **Monitoring Setup**: `vision-inference-terraform-setup.yaml` - Monitoring infrastructure
- **Retraining Monitor**: `vision-inference-retrain-decoy.yaml` - Low confidence monitoring

### Key URLs
- **GCP Console**: https://console.cloud.google.com/?project=medscanai-476500
- **Monitoring Dashboards**: https://console.cloud.google.com/monitoring/dashboards?project=medscanai-476500
- **Alert Policies**: https://console.cloud.google.com/monitoring/alerting?project=medscanai-476500
- **Cloud Run**: https://console.cloud.google.com/run?project=medscanai-476500
