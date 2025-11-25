# Dockerfile for Medical Image Classification Model Training
# Uses Python 3.9 as base image

FROM python:3.9

# Set working directory
WORKDIR /app

# Set environment variables early (for better caching)
ENV PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    MLFLOW_TRACKING_URI=file:///app/mlruns \
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

# Batch 3: ML libraries (moderate size)
RUN pip install --no-cache-dir scikit-learn matplotlib "seaborn>=0.12.0"

# Batch 4: MLflow and tuning (moderate size)
RUN pip install --no-cache-dir mlflow "mlflow[tensorflow]" keras-tuner

# Batch 5: Fairness and bias detection
RUN pip install --no-cache-dir fairlearn

# Batch 6: Model interpretability (SHAP and LIME)
RUN pip install --no-cache-dir shap lime scikit-image

# Copy training scripts and modules (this layer changes frequently)
COPY train_resnet.py train_vit.py train_custom_cnn.py bias_detection.py bias_mitigation.py metrics_tracker.py visualization.py lightweight_metrics_tracker.py interpretability.py ./

# Create directories in one layer
RUN mkdir -p ./config /app/data/preprocessed /app/models /app/logs

# CPU limit configuration
# Note: CPU limits are set at runtime using --cpus flag
# To limit to 50% of one CPU core, use: docker run --cpus="0.5" ...
# To limit to 50% of available CPUs, calculate based on your system

# Expose port for MLflow UI (optional)
EXPOSE 5000

# Default command (can be overridden to use train_resnet.py, train_vit.py, or train_custom_cnn.py)
CMD ["python", "train_resnet.py", "--config", "/app/config/vision_training.yml", "--data_path", "/app/data/preprocessed", "--output_path", "/app/data"]

