# How to Actually Delete Vision Inference Resources

## ⚠️ Why Terraform Destroy Didn't Work

Your Terraform destroy command showed:
```
No changes. No objects need to be destroyed.
```

**Reason:** Your resources were created manually using `gcloud` commands, **not** by Terraform. Terraform has no state tracking these resources, so it can't delete them.

---

## 🎯 Three Ways to Delete Resources

### Option 0: GitHub Actions Workflow (Easiest - Now Fixed!)

The workflow now uses a smart cleanup script that handles both cases:

```
1. Go to: Actions → Vision Inference - Terraform Setup
2. Click: Run workflow
3. Select: action=destroy, auto_approve=true
4. Click: Run workflow
```

**What happens:**
- ✅ Tries Terraform destroy first
- ✅ Detects manually created resources
- ✅ Deletes them via gcloud automatically
- ✅ Verifies complete cleanup

**This should now work correctly!** 🎉

---

### Option 1: Manual Deletion (Quickest)

Delete resources using gcloud commands:

```bash
cd deploymentVisionInference/terraform
chmod +x manual_delete.sh
./manual_delete.sh
```

This script will delete:
- ❌ Cloud Run service: `vision-inference-api`
- ❌ Artifact Registry: `vision-inference`

**What's NOT deleted** (safe):
- ✅ GCS bucket and models
- ✅ Other Cloud Run services (rag-service)
- ✅ IAM permissions (harmless to leave)

---

### Option 2: Import Then Destroy (Recommended for Future)

If you want Terraform to manage these resources going forward:

#### Step 1: Import existing resources

```bash
cd deploymentVisionInference/terraform
chmod +x import_existing.sh
./import_existing.sh
```

This tells Terraform about your existing resources.

#### Step 2: Verify import

```bash
terraform plan
```

Should show "No changes" (Terraform now knows about everything).

#### Step 3: Now destroy via Terraform

```bash
# Local destroy
terraform destroy

# Or via GitHub Actions
# Go to: Actions → Vision Inference - Terraform Setup
# Select: action=destroy, auto_approve=true
```

---

## 📋 Manual Commands (If Scripts Don't Work)

### Delete Cloud Run Service
```bash
gcloud run services delete vision-inference-api \
  --region=us-central1 \
  --project=medscanai-476500 \
  --quiet
```

### Delete Artifact Registry
```bash
gcloud artifacts repositories delete vision-inference \
  --location=us-central1 \
  --project=medscanai-476500 \
  --quiet
```

### Remove IAM Bindings (Optional)
```bash
# Remove Cloud Build Run Admin
gcloud projects remove-iam-policy-binding medscanai-476500 \
  --member="serviceAccount:246542889931-compute@developer.gserviceaccount.com" \
  --role="roles/run.admin"

# Remove Service Account User
gcloud projects remove-iam-policy-binding medscanai-476500 \
  --member="serviceAccount:246542889931-compute@developer.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"
```

---

## 🔄 Going Forward: Use Terraform Properly

### For First-Time Setup:

**✅ DO THIS (Terraform creates resources):**
```bash
# Via GitHub Actions
Actions → Vision Inference - Terraform Setup → action=apply
```

**❌ DON'T DO THIS (Manual gcloud):**
```bash
# Don't manually create resources
gcloud run deploy vision-inference-api ...  # ❌
```

### Why?

If Terraform creates the resources, it tracks them in its state. Then:
- `terraform plan` - see changes
- `terraform apply` - update resources
- `terraform destroy` - **actually deletes them** ✅

---

## 📊 Check What Actually Exists

```bash
# Check Cloud Run services
gcloud run services list --region=us-central1 --project=medscanai-476500

# Check Artifact Registry
gcloud artifacts repositories list --location=us-central1 --project=medscanai-476500

# Check if vision-inference-api exists
gcloud run services describe vision-inference-api \
  --region=us-central1 \
  --project=medscanai-476500 2>&1 | head -5
```

---

## ✅ Verification After Deletion

After running deletion commands:

```bash
# Should return "NOT_FOUND" or similar error
gcloud run services describe vision-inference-api \
  --region=us-central1 \
  --project=medscanai-476500

# Should NOT list vision-inference
gcloud artifacts repositories list \
  --location=us-central1 \
  --project=medscanai-476500
```

---

## 🆘 Troubleshooting

### "Permission denied"
You need these roles:
- `roles/run.admin`
- `roles/artifactregistry.admin`

Contact your GCP project admin or owner.

### "Service not found"
Resources might already be deleted! Check with:
```bash
gcloud run services list --region=us-central1 --project=medscanai-476500
```

### "Cannot delete repository - contains images"
Force delete:
```bash
gcloud artifacts repositories delete vision-inference \
  --location=us-central1 \
  --project=medscanai-476500 \
  --quiet \
  --force
```

---

## 📚 Summary

| Method | When to Use | Pros | Cons |
|--------|-------------|------|------|
| **Manual gcloud** | Quick deletion | Fast, simple | Terraform doesn't know |
| **Import + Terraform** | Want Terraform management | Future control | Extra step |

**Recommendation:** Use manual deletion now, then use Terraform properly for future deployments.
