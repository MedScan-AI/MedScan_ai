# Deployment Steps Summary

## Commands Executed

All gcloud command outputs have been saved to files in this temp directory.

### Step 1: gcloud Version Check
**File:** `step1.txt`
**Command:**
```powershell
gcloud --version
```

### Step 2: Project Check
**File:** `step2.txt`
**Command:**
```powershell
gcloud config get-value project
```

### Step 3: Authentication Check
**File:** `step3_auth.txt`
**Command:**
```powershell
gcloud auth list
```

### Step 4: Existing Services
**File:** `step4_existing_services.txt`
**Command:**
```powershell
gcloud run services list --region=us-central1 --project=medscanai-476500
```

### Step 5: Recent Builds
**File:** `step5_recent_builds.txt`
**Command:**
```powershell
gcloud builds list --limit=3 --project=medscanai-476500 --region=us-central1
```

### Step 8: Build Submission
**File:** `step8_build.txt`
**Command:**
```powershell
gcloud builds submit --config=ModelDevelopment/VisionInference/cloudbuild.yaml --substitutions=_SERVICE_NAME=vision-inference-api,_REGION=us-central1 --project=medscanai-476500 --region=us-central1
```

### Step 9: Build Status
**File:** `step9_status.txt`
**Command:**
```powershell
gcloud builds list --limit=1 --project=medscanai-476500 --region=us-central1
```

### Step 10: Service URL
**File:** `step10_url.txt`
**Command:**
```powershell
gcloud run services describe vision-inference-api --region=us-central1 --project=medscanai-476500 --format="value(status.url)"
```

## How to Check Results

1. **View all temp files:**
   ```powershell
   Get-ChildItem "C:\Users\sriha\NEU\MLOPS\workspace2\MedScan_ai\ModelDevelopment\VisionInference\temp"
   ```

2. **Read a specific file:**
   ```powershell
   Get-Content "C:\Users\sriha\NEU\MLOPS\workspace2\MedScan_ai\ModelDevelopment\VisionInference\temp\step8_build.txt"
   ```

3. **Check build status:**
   ```powershell
   gcloud builds list --limit=1 --project=medscanai-476500 --region=us-central1
   ```

4. **Check service status:**
   ```powershell
   gcloud run services describe vision-inference-api --region=us-central1 --project=medscanai-476500
   ```

5. **Get service URL:**
   ```powershell
   gcloud run services describe vision-inference-api --region=us-central1 --project=medscanai-476500 --format="value(status.url)"
   ```

## Cloud Console Links

- **Cloud Build Console:** https://console.cloud.google.com/cloud-build/builds?project=medscanai-476500
- **Cloud Run Console:** https://console.cloud.google.com/run?project=medscanai-476500

## Next Steps

1. Check the build status in Cloud Build console
2. Wait for build to complete (typically 5-10 minutes)
3. Once deployed, test the service:
   ```powershell
   $url = gcloud run services describe vision-inference-api --region=us-central1 --project=medscanai-476500 --format="value(status.url)"
   Invoke-WebRequest -Uri "$url/health"
   ```
