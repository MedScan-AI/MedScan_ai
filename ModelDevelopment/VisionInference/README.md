# Vision Inference Pipeline

This directory contains the inference pipeline for Vision models (TB and Lung Cancer detection using ResNet models).

## Quick Links

- **[GCP Setup Guide](./GCP_SETUP.md)** - How to set up GCP credentials on your local system
- **[Deployment Guide](./DEPLOYMENT.md)** - Deploy to GCP Cloud Run as a serverless HTTP endpoint

## Overview

The inference pipeline provides REST API endpoints using FastAPI for:
- **TB Detection**: `/predict/tb` - Predicts Tuberculosis from chest X-ray images
- **Lung Cancer Detection**: `/predict/lung_cancer` - Predicts Lung Cancer from CT scan images

**API Documentation**: FastAPI automatically generates interactive API documentation available at:
- Swagger UI: `http://localhost:5000/docs`
- ReDoc: `http://localhost:5000/redoc`

## Model Location

The pipeline automatically finds and loads the latest trained models from:
```
ModelDevelopment/data/models/YYYY/MM/DD/timestamp/{dataset_name}_CNN_ResNet18/
```

Where:
- `{dataset_name}` is either `tb` or `lung_cancer_ct_scan`
- Models are stored as `.keras` files (e.g., `CNN_ResNet18_best.keras`)

The pipeline selects the latest timestamp for each model type.

## API Endpoints

### Health Check
```
GET /health
```
Returns the health status and model loading status.

**Response:**
```json
{
  "status": "healthy",
  "tb_model_loaded": true,
  "lung_cancer_model_loaded": true
}
```

### TB Prediction
```
POST /predict/tb
```
Predicts TB from a chest X-ray image.

**Request Format:**
- Multipart form data (file upload):
```
Content-Type: multipart/form-data
file: <file>
```

**Response:**
```json
{
  "status": "success",
  "model": "tb_CNN_ResNet18",
  "model_path": "/path/to/model.keras",
  "predicted_class": "Tuberculosis",
  "confidence": 0.95,
  "class_probabilities": {
    "Normal": 0.05,
    "Tuberculosis": 0.95
  },
  "gradcam_image": "iVBORw0KGgoAAAANSUhEUgAA..." // Base64 encoded PNG image
}
```

**Note:** The `gradcam_image` field contains a base64-encoded PNG image showing:
- **Left panel**: Original image
- **Middle panel**: GradCAM heatmap showing model attention
- **Right panel**: Overlay of original image and heatmap

### Lung Cancer Prediction
```
POST /predict/lung_cancer
```
Predicts Lung Cancer from a CT scan image.

**Request/Response:** Same format as TB endpoint.

## Running Locally

### Prerequisites
- Python 3.9+
- Trained models in `ModelDevelopment/data/models/`

### Setup
```bash
cd ModelDevelopment/VisionInference
pip install -r requirements.txt
```

### Run

#### Linux/macOS (bash/zsh)
```bash
# Set models path (optional, defaults to ../data/models)
export MODELS_BASE_PATH=/path/to/models

# Run server with uvicorn
uvicorn app:app --host 0.0.0.0 --port 5000

# Or run directly
python app.py
```

#### Windows PowerShell
```powershell
# Set models path (optional, defaults to ../data/models)
$env:MODELS_BASE_PATH = "C:\path\to\models"

# Run server with uvicorn
uvicorn app:app --host 0.0.0.0 --port 5000

# Or run directly
python app.py
```

#### Windows Command Prompt (cmd)
```cmd
REM Set models path (optional, defaults to ../data/models)
set MODELS_BASE_PATH=C:\path\to\models

REM Run server with uvicorn
uvicorn app:app --host 0.0.0.0 --port 5000

REM Or run directly
python app.py
```

The server will start on `http://localhost:5000`
- API Documentation: `http://localhost:5000/docs`
- ReDoc: `http://localhost:5000/redoc`

## Docker

### Build

**Linux/macOS (bash/zsh):**
```bash
docker build -t vision-inference:latest .
```

**Windows PowerShell:**
```powershell
docker build -t vision-inference:latest .
```

**Windows Command Prompt (cmd):**
```cmd
docker build -t vision-inference:latest .
```

### Run

#### Linux/macOS (bash/zsh)

**Absolute path:**
```bash
docker run -p 5000:5000 \
  -v /absolute/path/to/ModelDevelopment/data/models:/app/data/models \
  vision-inference:latest
```

**Relative path (from project root):**
```bash
docker run -p 5000:5000 \
  -v "$(pwd)/../data/models:/app/data/models" \
  vision-inference:latest
```

#### Windows PowerShell

**Absolute path:**
```powershell
docker run -p 5000:5000 `
  -v C:\Users\sriha\NEU\MLOPS\workspace2\MedScan_ai\ModelDevelopment\data\models:/app/data/models `
  vision-inference:latest
```

**Relative path (from project root):**
```powershell
docker run -p 5000:5000 `
  -v "${PWD}\..\data\models:/app/data/models" `
  vision-inference:latest
```

**Note:** In PowerShell, use backticks (`) for line continuation and forward slashes (/) in the container path.

#### Windows Command Prompt (cmd)

**Absolute path:**
```cmd
docker run -p 5000:5000 -v C:\Users\sriha\NEU\MLOPS\workspace2\MedScan_ai\ModelDevelopment\data\models:/app/data/models vision-inference:latest
```

**Relative path:**
```cmd
docker run -p 5000:5000 -v %CD%\..\data\models:/app/data/models vision-inference:latest
```

**Note:** In cmd, use a single line or `^` for line continuation. Use forward slashes (/) in the container path.

### Docker Compose

**Linux/macOS (bash/zsh):**
```bash
docker-compose up
# Or in detached mode
docker-compose up -d
```

**Windows PowerShell:**
```powershell
docker-compose up
# Or in detached mode
docker-compose up -d
```

**Windows Command Prompt (cmd):**
```cmd
docker-compose up
REM Or in detached mode
docker-compose up -d
```

The `docker-compose.yml` file is already configured with the correct volume mounts.

## Environment Variables

- `MODELS_BASE_PATH`: Base path to models directory (default: `ModelDevelopment/data/models`)
- `PORT`: Server port (default: `5000`)
- `HOST`: Server host (default: `0.0.0.0`)

## Testing

### Using curl

#### Linux/macOS (bash/zsh)

**Health check:**
```bash
curl http://localhost:5000/health
```

**TB prediction (with file upload):**
```bash
curl -X POST http://localhost:5000/predict/tb \
  -F "file=@/path/to/xray.jpg"
```

**Lung Cancer prediction (with file upload):**
```bash
curl -X POST http://localhost:5000/predict/lung_cancer \
  -F "file=@/path/to/ct_scan.jpg"
```

#### Windows PowerShell

**Health check:**
```powershell
curl http://localhost:5000/health
# Or use Invoke-WebRequest
Invoke-WebRequest -Uri http://localhost:5000/health
```

**TB prediction (with file upload):**
```powershell
curl.exe -X POST http://localhost:5000/predict/tb `
  -F "file=@C:\path\to\xray.jpg"
```

**Lung Cancer prediction (with file upload):**
```powershell
curl.exe -X POST http://localhost:5000/predict/lung_cancer `
  -F "file=@C:\path\to\ct_scan.jpg"
```

#### Windows Command Prompt (cmd)

**Health check:**
```cmd
curl http://localhost:5000/health
```

**TB prediction (with file upload):**
```cmd
curl -X POST http://localhost:5000/predict/tb -F "file=@C:\path\to\xray.jpg"
```

**Lung Cancer prediction (with file upload):**
```cmd
curl -X POST http://localhost:5000/predict/lung_cancer -F "file=@C:\path\to\ct_scan.jpg"
```

### Using Python
```python
import requests

# File upload
with open("xray.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post(
        "http://localhost:5000/predict/tb",
        files=files
    )
    print(response.json())
```

## Notes

- Models are loaded at startup. If models are not found, the endpoints will return an error.
- The pipeline expects models to be in `.keras` format (TensorFlow/Keras saved model format).
- Image preprocessing includes resizing to 224x224 and normalization to [0, 1].
- Class names are automatically loaded from training metadata if available, otherwise defaults are used.
- FastAPI provides automatic API documentation at `/docs` and `/redoc` endpoints.
- The API uses async/await for better performance (though model inference itself is synchronous).
