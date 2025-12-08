# Dockerfile for Vision Model Inference Pipeline
# Uses Python 3.9 as base image (matching training Dockerfile)

FROM python:3.9

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies (including OpenCV requirements)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgthread-2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip
RUN pip install --upgrade "pip<24.1"

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy inference application files
COPY app.py model_loader.py gradcam.py ./

# Create directory for models
RUN mkdir -p /app/data/models

# Note: Models will be loaded from GCS at runtime
# Set environment variable to use GCS
ENV USE_GCS_MODELS=true \
    GCS_BUCKET_NAME=medscan-pipeline-medscanai-476500 \
    GCS_MODELS_PREFIX=vision/trained_models

# Expose port for Flask API
EXPOSE 5000

# Set default command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
