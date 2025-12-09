# Smart Cleanup Script

## 📜 Overview

`cleanup_resources.sh` - An intelligent cleanup script that handles Vision Inference resource deletion, whether resources were created by Terraform or manually.

## 🎯 Problem It Solves

### The Issue:
- Resources created manually (via `gcloud` or console) → Terraform doesn't know about them
- `terraform destroy` shows: "0 resources destroyed"
- Resources still exist in GCP, costing money

### The Solution:
This script:
1. ✅ Tries Terraform destroy first (for Terraform-managed resources)
2. ✅ Detects if resources still exist (manually created)
3. ✅ Deletes remaining resources using gcloud
4. ✅ Verifies complete cleanup

## 🚀 Usage

### Via GitHub Actions (Easiest)

```
Actions → Vision Inference - Terraform Setup → Run workflow
- action: destroy
- auto_approve: true
```

### Via Terminal

```bash
cd deploymentVisionInference/terraform

# Set environment variables (optional - defaults provided)
export PROJECT_ID="medscanai-476500"
export REGION="us-central1"
export SERVICE_NAME="vision-inference-api"
export REPO_NAME="vision-inference"

# Run script
chmod +x cleanup_resources.sh
./cleanup_resources.sh
```

## 🔍 What It Does

### Step 1: Terraform Destroy
```
Runs: terraform destroy -auto-approve
Checks: Did it actually destroy resources?
Result: Tracks if Terraform managed anything
```

### Step 2: Check for Remaining Resources
```
Checks: Cloud Run service exists?
Checks: Artifact Registry exists?
```

### Step 3: Manual Cleanup (if needed)
```
Deletes: Cloud Run service via gcloud
Deletes: Artifact Registry via gcloud
```

### Step 4: Verification
```
Confirms: All resources removed
Reports: Success or warnings
```

## 📊 Output Examples

### Case 1: Terraform-Managed Resources

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1: Attempting Terraform destroy...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Terraform destroy completed
✅ Terraform successfully destroyed resources

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 2: Checking for manually created resources...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Checking Cloud Run service: vision-inference-api...
  ✅ Service does not exist (already deleted)

Checking Artifact Registry: vision-inference...
  ✅ Repository does not exist (already deleted)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ All resources cleaned up successfully
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Case 2: Manually Created Resources

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1: Attempting Terraform destroy...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Terraform destroy completed
⚠️  Terraform destroyed 0 resources - resources may have been created manually

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 2: Checking for manually created resources...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Checking Cloud Run service: vision-inference-api...
  ⚠️  Service still exists - will delete manually
  Deleting Cloud Run service...
  ✅ Service deleted

Checking Artifact Registry: vision-inference...
  ⚠️  Repository still exists - will delete manually
  Deleting Artifact Registry repository...
  ✅ Repository deleted

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  Manual cleanup was needed
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This means resources were created outside of Terraform.
For future deployments, use Terraform to create resources:
  Actions → Vision Inference - Terraform Setup → action=apply
```

## 🔐 Required Permissions

The script needs:
- `roles/run.admin` - Delete Cloud Run services
- `roles/artifactregistry.admin` - Delete repositories

These are typically granted to:
- Project Owner
- Project Editor
- Cloud Build service account (in GitHub Actions)

## 🛡️ Safety Features

### What It DELETES:
- ❌ Cloud Run service: `vision-inference-api`
- ❌ Artifact Registry: `vision-inference`

### What It PRESERVES:
- ✅ GCS bucket: `medscan-pipeline-medscanai-476500`
- ✅ Trained models in GCS
- ✅ Other Cloud Run services (e.g., `rag-service`)
- ✅ Other Artifact Registry repositories
- ✅ IAM permissions (safe to leave)

### Verification:
- Script exits with code 0 if successful
- Script exits with code 1 if resources still exist
- Always checks resource status before/after

## 🐛 Troubleshooting

### "Permission denied"

**Cause:** Missing IAM roles

**Solution:**
```bash
# Check your permissions
gcloud projects get-iam-policy medscanai-476500 \
  --flatten="bindings[].members" \
  --filter="bindings.members:user:YOUR_EMAIL"
```

Request `roles/run.admin` and `roles/artifactregistry.admin` from project owner.

### "Service/Repository not found"

**Cause:** Already deleted

**Solution:** This is actually success! The script will still verify and report success.

### Script hangs

**Cause:** Waiting for user input

**Solution:** Make sure you're not running in a mode that expects confirmation. The script uses `--quiet` flags.

## 📚 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PROJECT_ID` | `medscanai-476500` | GCP project ID |
| `REGION` | `us-central1` | GCP region |
| `SERVICE_NAME` | `vision-inference-api` | Cloud Run service name |
| `REPO_NAME` | `vision-inference` | Artifact Registry name |

Override if using different names:

```bash
export PROJECT_ID="my-project"
export SERVICE_NAME="my-service"
./cleanup_resources.sh
```

## 🔄 Integration with Terraform Workflow

The script is automatically called by:
- `.github/workflows/vision-inference-terraform-setup.yaml`
- Trigger: `action=destroy` with `auto_approve=true`

Workflow steps:
1. Checkout code
2. Setup gcloud credentials
3. Initialize Terraform
4. **Run cleanup_resources.sh** ← This script
5. Report results

## 💡 Best Practices

### ✅ DO:
- Use this script for cleanup (both manual and Terraform resources)
- Run from GitHub Actions for automated cleanup
- Check output logs for verification
- Use Terraform for future infrastructure creation

### ❌ DON'T:
- Don't mix Terraform and manual resource creation
- Don't delete resources partially (use this script)
- Don't skip verification step

## 🎓 How It Works

```bash
# Pseudo-code logic

if terraform_destroy_succeeds and destroys_resources:
    # Terraform managed everything
    return "Success - Terraform cleanup"
    
elif terraform_destroy_succeeds but destroys_0_resources:
    # Resources were created manually
    check_if_resources_exist()
    
    if resources_exist:
        delete_with_gcloud()
        verify_deletion()
        return "Success - Manual cleanup"
    else:
        return "Success - Already deleted"
        
else:
    # Terraform failed
    try_manual_cleanup()
    return "Completed with warnings"
```

## 📈 Future Improvements

Potential enhancements:
- [ ] Dry-run mode (preview without deletion)
- [ ] Backup resource configurations before deletion
- [ ] Support for custom resource names via config file
- [ ] Email notification on completion
- [ ] Slack/Discord webhook integration

## 🔗 Related Files

- `main.tf` - Terraform resource definitions
- `versions.tf` - Terraform provider versions
- `variables.tf` - Terraform variables
- `README.md` - Main Terraform documentation
- `DESTROY_RESOURCES.md` - Detailed deletion guide
- `.github/workflows/vision-inference-terraform-setup.yaml` - Workflow using this script

## 📝 License

Part of the MedScan AI project infrastructure.
