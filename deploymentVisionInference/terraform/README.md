# Vision Inference API - Terraform Infrastructure

This directory contains Terraform configuration to deploy the Vision Inference API infrastructure on Google Cloud Platform.

## 📋 Prerequisites

1. **Terraform installed** (>= 1.0)
   ```bash
   terraform --version
   ```

2. **Google Cloud SDK (gcloud) installed and authenticated**
   ```bash
   gcloud auth application-default login
   ```

3. **GCP Project with billing enabled**

4. **Docker image built and pushed to Artifact Registry** (or will be built by Cloud Build)

## 🏗️ Infrastructure Components

This Terraform configuration creates:

- ✅ **Artifact Registry repository** for Docker images
- ✅ **Cloud Run service** for the inference API
- ✅ **IAM permissions** for Cloud Build and Cloud Run
- ✅ **Service APIs** (Cloud Run, Artifact Registry, Cloud Build, Storage)
- ✅ **Public access** (if enabled)

## 🚀 Quick Start

### 1. Initialize Terraform

```bash
cd ModelDevelopment/VisionInference/terraform
terraform init
```

### 2. Review and customize variables

Copy the example variables file:
```bash
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars` as needed:
```hcl
project_id = "your-project-id"
region     = "us-central1"
memory     = "2Gi"
cpu        = "1"
```

### 3. Plan the deployment

```bash
terraform plan
```

Review the changes that will be made.

### 4. Apply the configuration

```bash
terraform apply
```

Type `yes` when prompted.

### 5. Get the service URL

```bash
terraform output service_url
```

## 📝 Configuration Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `project_id` | GCP Project ID | `medscanai-476500` |
| `region` | GCP region | `us-central1` |
| `service_name` | Cloud Run service name | `vision-inference-api` |
| `repository_name` | Artifact Registry repo | `vision-inference` |
| `gcs_bucket_name` | GCS bucket for models | `medscan-pipeline-medscanai-476500` |
| `memory` | Memory allocation | `2Gi` |
| `cpu` | CPU allocation | `1` |
| `min_instances` | Min instances (0 = scale to zero) | `0` |
| `max_instances` | Max instances | `3` |
| `timeout` | Request timeout (seconds) | `300` |
| `allow_unauthenticated` | Allow public access | `true` |

## 🔧 Usage Examples

### Deploy with custom resources

```bash
terraform apply \
  -var="memory=4Gi" \
  -var="cpu=2" \
  -var="max_instances=5"
```

### Deploy to a different project

```bash
terraform apply \
  -var="project_id=my-other-project" \
  -var="region=us-east1"
```

### Disable public access

```bash
terraform apply -var="allow_unauthenticated=false"
```

## 📤 Outputs

After deployment, Terraform provides these outputs:

```bash
# Service URL
terraform output service_url

# Health check endpoint
terraform output health_endpoint

# API docs
terraform output api_docs_endpoint

# Prediction endpoints
terraform output tb_prediction_endpoint
terraform output lung_cancer_prediction_endpoint
```

## 🔄 Updating the Infrastructure

After making changes to `main.tf` or variables:

```bash
terraform plan
terraform apply
```

## 🗑️ Destroying Resources

To remove all created resources:

```bash
terraform destroy
```

⚠️ **Warning**: This will delete the Cloud Run service and Artifact Registry repository. Your Docker images and GCS models will remain.

## 📦 Building and Deploying the Container

Terraform creates the infrastructure but doesn't build/push the Docker image. Use Cloud Build:

```bash
cd ../..  # Go to project root
gcloud builds submit \
  --config=ModelDevelopment/VisionInference/cloudbuild.yaml \
  --project=medscanai-476500 \
  --region=us-central1
```

## 🔍 Verifying Deployment

After Terraform completes:

1. **Check service status**:
   ```bash
   gcloud run services describe vision-inference-api \
     --region=us-central1 \
     --project=medscanai-476500
   ```

2. **Test health endpoint**:
   ```bash
   curl $(terraform output -raw service_url)/health
   ```

3. **View API docs**:
   ```bash
   open $(terraform output -raw api_docs_endpoint)
   ```

## 🐛 Troubleshooting

### "Resource already exists" error

If resources already exist, import them:
```bash
terraform import google_cloud_run_service.vision_inference_api \
  locations/us-central1/namespaces/medscanai-476500/services/vision-inference-api
```

### "Permission denied" errors

Ensure you have necessary IAM roles:
- `roles/owner` or
- `roles/editor` or
- Specific roles: `roles/run.admin`, `roles/artifactregistry.admin`, `roles/iam.securityAdmin`

### Service won't start

Check logs:
```bash
gcloud run services logs read vision-inference-api \
  --region=us-central1 \
  --project=medscanai-476500 \
  --limit=50
```

## 📚 Additional Resources

- [Terraform Google Provider Docs](https://registry.terraform.io/providers/hashicorp/google/latest/docs)
- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Artifact Registry Documentation](https://cloud.google.com/artifact-registry/docs)

## 💰 Cost Estimation

With default settings (1 CPU, 2GB RAM, scale to zero):
- **Idle**: $0 (scales to zero)
- **Active**: ~$0.00004 per second
- **Monthly (light usage)**: ~$0.10 - $2.00

Use [GCP Pricing Calculator](https://cloud.google.com/products/calculator) for detailed estimates.
