# Quick Delete Commands

## 🚨 Delete Artifact Registry NOW

Run these commands in your terminal:

### Option 1: Simple Delete

```bash
gcloud artifacts repositories delete vision-inference \
  --location=us-central1 \
  --project=medscanai-476500 \
  --quiet
```

### Option 2: If Option 1 Fails

```bash
gcloud artifacts repositories delete \
  projects/medscanai-476500/locations/us-central1/repositories/vision-inference \
  --quiet
```

### Option 3: Using the Script

```bash
cd deploymentVisionInference/terraform
chmod +x delete_artifact_registry.sh
./delete_artifact_registry.sh
```

---

## ✅ Verify Deletion

After running the delete command, verify it's gone:

```bash
# Should return error or empty list
gcloud artifacts repositories list \
  --location=us-central1 \
  --project=medscanai-476500 | grep vision-inference
```

If the above returns nothing, the repository is deleted! ✅

---

## 🔍 Check What's Inside (Before Deleting)

To see what images are in the repository:

```bash
gcloud artifacts docker images list \
  us-central1-docker.pkg.dev/medscanai-476500/vision-inference \
  --format="table(package,version,create_time)"
```

---

## ❓ Why Didn't the Script Delete It?

Possible reasons:
1. **Images inside** - Repository with images requires explicit deletion
2. **Permissions** - You might lack `artifactregistry.repositories.delete` permission
3. **Script error** - Command failed silently

The updated cleanup script now handles these cases better.

---

## 🆘 If Still Can't Delete

### Check Permissions

```bash
# Check your permissions
gcloud projects get-iam-policy medscanai-476500 \
  --flatten="bindings[].members" \
  --filter="bindings.members:user:$(gcloud config get-value account)" \
  --format="table(bindings.role)"
```

You need one of these roles:
- `roles/owner`
- `roles/editor`
- `roles/artifactregistry.admin`
- `roles/artifactregistry.repoAdmin`

### Force Delete via Console

1. Go to: https://console.cloud.google.com/artifacts?project=medscanai-476500
2. Find: `vision-inference` repository
3. Click: ⋮ (three dots) → Delete
4. Confirm deletion

---

## 💰 Cost Note

Artifact Registry charges for storage:
- **~$0.10/GB/month** for stored images

If you're not using it, delete it to avoid charges!
