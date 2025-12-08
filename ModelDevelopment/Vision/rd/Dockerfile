# Dockerfile for ResNet18 Model Training (Simplified)
# Uses Python 3.9 as base image

FROM python:3.9

# Set working directory
WORKDIR /app

# Set environment variables early (for better caching)
ENV PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip first (cached layer)
RUN pip install --upgrade "pip<24.1"

# Copy requirements file (this layer is cached if requirements.txt doesn't change)
COPY requirements.txt .

# Install Python dependencies in optimized batches for better layer caching
# Batch 1: Core dependencies (rarely change, cached well)
RUN pip install --no-cache-dir numpy pandas Pillow pyyaml

# Batch 2: TensorFlow (heavy, stable, good for caching)
RUN pip install --no-cache-dir tensorflow

# Copy training script and config (this layer changes frequently)
COPY resnet.py config.yml ./

# Create directories in one layer
RUN mkdir -p ./config /app/data/preprocessed /app/data/models /app/data/logs

# Default command
CMD ["python", "resnet.py", "--config", "/app/config.yml", "--data_path", "/app/data/preprocessed", "--output_path", "/app/data"]
