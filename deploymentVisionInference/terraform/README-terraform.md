# Vision Inference - Terraform Infrastructure Setup

This workflow provisions the GCP infrastructure for the Vision Inference API using Terraform.

## 🎯 Purpose

**One-time infrastructure setup** that creates:
- ✅ Artifact Registry repository (for Docker images)
- ✅ Cloud Run service (initial setup)
- ✅ IAM permissions (Cloud Build, Cloud Run)
- ✅ Enabled APIs (Run, Artifact Registry, Cloud Build, Storage)

## 🔄 When to Use

### First Time Setup
Run this workflow **before** the application deployment workflow when:
- Infrastructure doesn't exist yet
- Setting up a new environment
- Creating resources from scratch

### Infrastructure Changes
Run when modifying:
- Resource configurations (CPU, memory, scaling)
- IAM permissions
- Environment variables
- Infrastructure-as-Code definitions

### Infrastructure Destruction
Run with `destroy` action to:
- Clean up all resources
- Remove the service and repository
- Delete infrastructure (keeps GCS bucket/models)

## 🚀 Usage

### Option 1: Manual Trigger (Recommended for First Time)

1. **Go to GitHub Actions:**
   ```
   Actions → Vision Inference - Terraform Setup → Run workflow
   ```

2. **Select action:**
   - **`plan`** - Preview changes (safe, no modifications)
   - **`apply`** - Create/update infrastructure
   - **`destroy`** - Delete all resources

3. **Auto-approve:**
   - ✅ Check for `apply`/`destroy` without manual confirmation
   - ⬜ Leave unchecked to see plan first

### Option 2: Automatic Trigger

Push changes to Terraform files:
```bash
# Modify infrastructure
cd ModelDevelopment/VisionInference/terraform
vim main.tf

# Commit and push
git add terraform/
git commit -m "Update infrastructure config"
git push origin main

# Workflow runs: terraform plan
# (Does NOT apply automatically from push)
```

## 📊 Workflow Actions

### Plan (Default)
```yaml
action: plan
auto_approve: false
```
**What it does:**
- ✅ Shows what will be created/changed/destroyed
- ✅ No actual changes made
- ✅ Safe to run anytime

**Use when:**
- Testing configuration changes
- Reviewing infrastructure updates
- Before applying changes

### Apply
```yaml
action: apply
auto_approve: true
```
**What it does:**
- ✅ Creates/updates infrastructure
- ✅ Applies Terraform configuration
- ✅ Sets up all resources

**Use when:**
- First-time setup
- Deploying infrastructure changes
- Scaling resources

> Note: The log-based metric `low_confidence_predictions` already exists in this project. By default `create_low_conf_metric=false` to avoid a 409 conflict. If you ever need Terraform to create it (only when it does not already exist), set `create_low_conf_metric=true` in `terraform.tfvars`.

### Destroy
```yaml
action: destroy
auto_approve: true
```
**What it does:**
- ❌ Deletes Cloud Run service
- ❌ Deletes Artifact Registry repository
- ⚠️ **WARNING: This removes the service!**

**Use when:**
- Tearing down environment
- Cleaning up test resources
- Removing infrastructure

## 🔐 Required Permissions

The GitHub Actions service account needs:
- `roles/run.admin` - Manage Cloud Run
- `roles/artifactregistry.admin` - Manage Artifact Registry
- `roles/iam.securityAdmin` - Grant IAM permissions
- `roles/storage.admin` - Manage GCS (for Terraform state)
- `roles/serviceusage.serviceUsageAdmin` - Enable APIs

## 📋 Workflow Steps

### 1. Pre-flight Checks
- ✅ Checkout code
- ✅ Authenticate to GCP
- ✅ Setup Terraform CLI

### 2. Validation
- ✅ Format check (`terraform fmt`)
- ✅ Initialize (`terraform init`)
- ✅ Validate configuration (`terraform validate`)

### 3. Plan
- ✅ Create execution plan (`terraform plan`)
- ✅ Display changes
- ✅ Comment on PR (if applicable)

### 4. Apply/Destroy
- ✅ Execute plan (if action=apply)
- ✅ Create/update resources
- ✅ Or destroy resources (if action=destroy)

### 5. Verification
- ✅ Get Terraform outputs
- ✅ Test deployed service
- ✅ Display summary

### 6. Artifacts
- ✅ Upload Terraform plan (plan action)
- ✅ Upload infrastructure state (apply action)

## 📤 Outputs

After successful apply, the workflow provides:

### Terraform Outputs
```
service_url                     = "https://vision-inference-api-123456.us-central1.run.app"
health_endpoint                 = "https://vision-inference-api-123456.us-central1.run.app/health"
api_docs_endpoint              = "https://vision-inference-api-123456.us-central1.run.app/docs"
tb_prediction_endpoint         = "https://vision-inference-api-123456.us-central1.run.app/predict/tb"
lung_cancer_prediction_endpoint = "https://vision-inference-api-123456.us-central1.run.app/predict/lung_cancer"
```

### Artifacts
- **Terraform Plan** (plan action) - 7 day retention
- **Infrastructure State** (apply action) - 90 day retention

## 🧪 Testing

After infrastructure is created, test manually:

```bash
# Get service URL from Terraform outputs
SERVICE_URL="https://vision-inference-api-246542889931.us-central1.run.app"

# Test health endpoint
curl $SERVICE_URL/health

# View API docs
open $SERVICE_URL/docs
```

**Note:** The service won't respond until the application is deployed using `vision-inference-deploy.yaml`.

## 🔄 Infrastructure Update Flow

```
1. Modify Terraform Config
   ↓
2. Run workflow with action=plan
   ↓
3. Review changes in workflow output
   ↓
4. Run workflow with action=apply + auto_approve=true
   ↓
5. Verify in GCP Console
```

## ⚠️ Important Notes

### Terraform State Management
- **State is stored locally** in the workflow (not persisted)
- For production, consider using [Terraform Cloud](https://cloud.hashicorp.com/products/terraform) or [GCS backend](https://www.terraform.io/docs/language/settings/backends/gcs.html)
- Current setup: State is recreated on each run (safe for declarative infrastructure)

### Cost Implications
- **Plan**: Free (no resources created)
- **Apply**: Creates billable resources
  - Cloud Run: $0 when scaled to zero
  - Artifact Registry: ~$0.10/GB/month
- **Destroy**: Stops all billing

### Idempotency
- Safe to run multiple times
- Terraform only changes what's different
- Re-running apply with same config = no changes

## 🐛 Troubleshooting

### "Error: already exists"
**Cause:** Resources already exist (created manually or by previous run)

**Solution:**
```bash
# Import existing resources
cd ModelDevelopment/VisionInference/terraform
terraform import google_cloud_run_service.vision_inference_api \
  locations/us-central1/namespaces/medscanai-476500/services/vision-inference-api
```

### "Permission denied"
**Cause:** Service account lacks necessary IAM roles

**Solution:** Check permissions in [Required Permissions](#-required-permissions)

### Plan shows unexpected changes
**Cause:** Manual changes made via gcloud/console

**Solution:**
- Review changes in plan output
- Either accept changes or update Terraform config to match current state

### Workflow timeout
**Cause:** Terraform operations taking too long

**Solution:**
- Check GCP Console for quota issues
- Retry workflow
- Contact GCP support if persistent

## 🔗 Related Workflows

| Workflow | Purpose | Relationship |
|----------|---------|--------------|
| **vision-inference-terraform-setup.yaml** | Infrastructure setup | **Run this first** |
| **vision-inference-deploy.yaml** | Application deployment | Depends on infrastructure |

## 📚 Additional Resources

- [Terraform GCP Provider Docs](https://registry.terraform.io/providers/hashicorp/google/latest/docs)
- [Cloud Run with Terraform](https://cloud.google.com/run/docs/deploying-source-code)
- [Terraform Workflows](https://www.terraform.io/docs/cloud/guides/recommended-practices/index.html)
- [Infrastructure as Code Best Practices](https://docs.microsoft.com/en-us/azure/architecture/framework/devops/iac)

## 💡 Best Practices

1. **Always run `plan` before `apply`**
   - Review changes
   - Understand impact
   - Catch errors early

2. **Use version control for Terraform config**
   - Commit all `.tf` files
   - Never commit `.tfstate` files
   - Use `.gitignore` (already configured)

3. **Document infrastructure changes**
   - Add comments to `.tf` files
   - Update commit messages
   - Link to related issues/PRs

4. **Test in stages**
   - Plan → Review → Apply
   - Don't use `auto_approve` unless certain
   - Verify in GCP Console after apply

5. **Use consistent naming**
   - Follow project conventions
   - Use descriptive resource names
   - Tag resources appropriately
