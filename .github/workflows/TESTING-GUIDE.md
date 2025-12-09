# Testing Guide for RAG Complete Setup Workflow

This guide explains how to test the `rag-complete-setup.yaml` workflow.

## Prerequisites

Before testing, ensure you have:

1. ✅ **GitHub Secrets configured:**
   - `GCP_SA_KEY`: Service account JSON with required permissions
   - `SMTP_USER` and `SMTP_PASSWORD` (optional, for notifications)

2. ✅ **GCP Resources ready:**
   - RAG model and index files in GCS: `gs://medscan-pipeline-medscanai-476500/RAG/`
   - HuggingFace token in Secret Manager: `huggingface-token`
   - Required GCP APIs enabled (Cloud Run, Cloud Build, Monitoring, Logging)

3. ✅ **Permissions:**
   - Service account has `roles/run.admin`
   - Service account has `roles/cloudbuild.builds.editor`
   - Service account has monitoring roles (workflow will grant these)

## Testing Methods

### Method 1: GitHub Actions UI (Recommended)

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

---

### Method 2: Push to Main Branch (Automatic Trigger)

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

---

### Method 3: Test Individual Components

#### Test Cloud Run Deployment Only

Use the existing `rag-deploy.yaml` workflow:
1. Go to Actions → "RAG Deployment"
2. Click "Run workflow"
3. This tests just the Cloud Run deployment part

#### Test Monitoring Infrastructure Only

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

---

## Verification Checklist

After running the workflow, verify:

### ✅ Cloud Run Service
- [ ] Service is deployed: `gcloud run services list --region=us-central1`
- [ ] Service URL is accessible
- [ ] Health endpoint returns `{"ready": true}`
- [ ] Config endpoint returns model configuration
- [ ] Prediction endpoint works (test with sample query)

### ✅ Monitoring Infrastructure
- [ ] Dashboard exists in GCP Console
- [ ] 7 custom metrics created (check Monitoring → Metrics)
- [ ] 11 alert policies created (check Monitoring → Alerting)
- [ ] Email notification channel created (if email provided)

### ✅ Permissions
- [ ] Service is publicly accessible (or IAM configured correctly)
- [ ] Service account has required permissions
- [ ] Monitoring permissions granted

---

## Testing Scenarios

### Scenario 1: First-Time Setup
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
- Alert policies created
- Email notifications configured

---

### Scenario 2: Service Update Only
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

### Scenario 3: Monitoring Setup Only
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

### Scenario 4: Error Handling
**Goal:** Test error handling

**Test Cases:**
1. **Missing GCP credentials:**
   - Remove or invalidate `GCP_SA_KEY` secret
   - Run workflow → Should fail with auth error

2. **Missing RAG data:**
   - Remove RAG model/index from GCS
   - Run workflow → Should fail when service tries to load model

3. **Terraform without auto-approve:**
   - Run with `auto_approve_terraform`: `false`
   - Workflow should stop at Terraform apply step
   - Should show instructions for manual approval

---

## Quick Test Commands

### Test Service Endpoints
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

### Check Monitoring Resources
```bash
# List custom metrics
gcloud logging metrics list --project=medscanai-476500 | grep rag_

# List alert policies
gcloud alpha monitoring policies list --project=medscanai-476500 | grep "RAG Service"

# List dashboards
gcloud monitoring dashboards list --project=medscanai-476500 | grep "RAG Service"
```

### Check Terraform State
```bash
cd deploymentRAG/terraform
terraform state list
terraform output
```

---

## Troubleshooting

### Workflow Fails at Cloud Run Deployment

**Symptoms:**
- Job `deploy-cloud-run` fails
- Cloud Build fails

**Solutions:**
1. Check Cloud Build logs in GCP Console
2. Verify RAG model/index exist in GCS
3. Check HuggingFace token in Secret Manager
4. Verify service account has `roles/run.admin`

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
cd deploymentRAG/terraform
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
gcloud logging metrics list --project=medscanai-476500 | grep rag_

# Check recent logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=rag-service AND jsonPayload.prediction_result:*" \
  --project=medscanai-476500 \
  --limit=10 \
  --format=json

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

## Best Practices for Testing

1. **Test in stages:**
   - First test Cloud Run deployment only
   - Then test monitoring setup separately
   - Finally test complete workflow

2. **Use test email:**
   - Use a test email address for notifications
   - Verify alerts are received

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

---

## Cleanup After Testing

If you want to clean up test resources:

```bash
# Delete Cloud Run service
gcloud run services delete rag-service \
  --region=us-central1 \
  --project=medscanai-476500

# Delete monitoring resources (via Terraform)
cd deploymentRAG/terraform
terraform destroy
```

**Note:** Be careful with cleanup - make sure you're not deleting production resources!

---

## Next Steps After Successful Test

1. ✅ Document any issues found
2. ✅ Update workflow if needed
3. ✅ Set up scheduled monitoring (`rag-monitoring.yaml`)
4. ✅ Configure production alerts
5. ✅ Test retraining workflow if needed

---

## Support

If you encounter issues:
1. Check workflow logs in GitHub Actions
2. Check Cloud Build logs in GCP Console
3. Check Cloud Run logs
4. Review Terraform state and outputs
5. Check this guide's troubleshooting section

