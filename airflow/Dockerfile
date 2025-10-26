FROM apache/airflow:2.10.1

USER root

# Install system dependencies for image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Copy requirements file
COPY requirements-pipeline.txt /opt/airflow/requirements-pipeline.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /opt/airflow/requirements-pipeline.txt

# Set environment variables for better Python behavior
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
