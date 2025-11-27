## How to Run the Vision Training Pipeline (Cloud Build → Vertex AI)

This guide shows how to verify your GCP setup and run the CI/CD pipeline that trains, validates, and deploys the Vision model.

### Prerequisites
- Google Cloud CLI installed and authenticated.
- Project set: `gcloud config set project medscanai-476500`
- Region: `us-central1`
- Bucket: `gs://medscan-pipeline-medscanai-476500`
- APIs enabled: Cloud Build, Vertex AI, Cloud Storage, Artifact Registry, Secret Manager (optional).
- The repository is checked out locally and you are in its root directory.

### 1) Verify GCP setup (recommended)
Run the verification script. It reads `airflow/.env` (if present) for `GCP_PROJECT_ID` and `GCS_BUCKET_NAME`.

```bash
bash scripts/verify_gcp_setup.sh
```

Confirm that:
- Project and bucket resolve correctly
- Required APIs are enabled
- You have access to Storage and Vertex AI

### 2) Run the pipeline (Cloud Build)
Submit the Cloud Build with dataset and epoch overrides. For a quick run, set epochs to 3.

```bash
gcloud builds submit \
  --config=cloudbuild/vision-training.yaml \
  --substitutions=_DATASET=tb,_EPOCHS=3 \
  --project=medscanai-476500 \
  --region=us-central1
```

Notes:
- Training, validation, bias check, and deployment run as separate steps.
- Each step logs to Cloud Build; open the build in the console to stream logs.
- The deployment step registers the model and deploys it to a Vertex AI endpoint (first-time deploys can take 30–60+ minutes).

### 3) Monitor
- Console: Cloud Build → History → select the running build.
- CLI (latest build): `gcloud builds list --project=medscanai-476500 --region=us-central1 --limit=1`

### 4) Where artifacts go
- Trained models and logs:
  - `gs://medscan-pipeline-medscanai-476500/vision/trained_models/${BUILD_ID}/`
- Validation results:
  - `gs://medscan-pipeline-medscanai-476500/vision/validation/${BUILD_ID}/validation_results.json`
- Vertex AI model registry and endpoint:
  - Models: https://console.cloud.google.com/vertex-ai/models?project=medscanai-476500
  - Endpoints: https://console.cloud.google.com/vertex-ai/endpoints?project=medscanai-476500

### 5) Deployment behavior
- The pipeline exports a TensorFlow SavedModel, uploads it to GCS, registers it in Vertex AI, then deploys to an endpoint.
- Deployment uses `sync=True`, so Step 6 waits until Vertex AI finishes the long‑running operation.

To check a running deployment operation (replace with the operation ID from logs):
```bash
gcloud ai operations describe OPERATION_ID \
  --project=medscanai-476500 --region=us-central1
```

To cancel (this will fail the build step):
```bash
gcloud ai operations cancel OPERATION_ID \
  --project=medscanai-476500 --region=us-central1
```

### 6) Quick test vs full training
- Fast iteration: set `_EPOCHS=2` in the command above.
- True “dry run” mode is supported by the training script via `--dry_run` (limits to ~64 images and 2 epochs). If needed for CI, add `--dry_run` to the training step in `cloudbuild/vision-training.yaml` and re-run; remove it later for full runs.

### 7) Common troubleshooting

- Vertex AI registration error: “no files in directory gs://.../model.keras”
  - Fixed by registering a SavedModel directory (pipeline already sets this). Ensure the registration `artifact_uri` points to a directory with a SavedModel, not a single `.keras` file.

- Permission error when registering/deploying:
  - Grant the Vertex service agent Storage read on the bucket:
    ```bash
    gcloud storage buckets add-iam-policy-binding gs://medscan-pipeline-medscanai-476500 \
      --member="serviceAccount:service-246542889931@gcp-sa-aiplatform.iam.gserviceaccount.com" \
      --role="roles/storage.objectViewer" \
      --project=medscanai-476500
    ```

- Deployment step appears stuck for a long time
  - First deploys can take 30–60+ minutes. It’s blocking by design (`sync=True`). If you need the pipeline to continue, switch to `sync=False` in `deploy.py` and log the operation ID.

- Validation or test data not found
  - Ensure the DVC step pulled preprocessed data and the dataset `_DATASET` exists under `DataPipeline/data/preprocessed` (Cloud Build will show counts). Rerun the data pipeline if empty.

### 8) Clean up (optional)
- Undeploy model from endpoint in Vertex AI console or via CLI.
- Remove GCS artifacts if you are done testing:
  ```bash
  gsutil -m rm -r gs://medscan-pipeline-medscanai-476500/vision/trained_models/${BUILD_ID}/
  gsutil -m rm -r gs://medscan-pipeline-medscanai-476500/vision/validation/${BUILD_ID}/
  ```




