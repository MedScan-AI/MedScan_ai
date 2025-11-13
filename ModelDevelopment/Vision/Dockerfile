# Dockerfile for Medical Image Classification Model Training
# Uses Python 3.9 as base image

FROM python:3.9

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy training scripts
COPY train_resnet.py train_vit.py train_custom_cnn.py ./

# Create config directory (config file will be mounted as volume or can be copied separately)
RUN mkdir -p ./config

# Create directories for data and models
RUN mkdir -p /app/data/preprocessed && \
    mkdir -p /app/models && \
    mkdir -p /app/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV MLFLOW_TRACKING_URI=file:///app/mlruns

# CPU limit configuration
# Note: CPU limits are set at runtime using --cpus flag
# To limit to 50% of one CPU core, use: docker run --cpus="0.5" ...
# To limit to 50% of available CPUs, calculate based on your system

# Expose port for MLflow UI (optional)
EXPOSE 5000

# Default command (can be overridden to use train_resnet.py, train_vit.py, or train_custom_cnn.py)
CMD ["python", "train_resnet.py", "--config", "/app/config/vision_training.yml", "--data_path", "/app/data/preprocessed", "--output_path", "/app/data"]

