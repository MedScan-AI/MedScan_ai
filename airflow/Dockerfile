# Use Python 3.10 for TFDV compatibility
FROM apache/airflow:2.10.1-python3.10

USER root

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    git \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Vision pipeline dependencies
COPY requirements-pipeline.txt /opt/airflow/requirements-pipeline.txt
RUN pip install --no-cache-dir -r /opt/airflow/requirements-pipeline.txt

# RAG pipeline dependencies
COPY DataPipeline/scripts/RAG/requirements.txt /opt/airflow/requirements-rag.txt
RUN pip install --no-cache-dir -r /opt/airflow/requirements-rag.txt

RUN pip install --no-cache-dir \
    sentence-transformers \
    faiss-cpu \
    google-cloud-storage \
    dvc[gs] \
    pyyaml \
    markdownify

# Install Playwright browsers
RUN playwright install chromium

# Download NLTK data
RUN python -c "import nltk; \
    nltk.download('punkt', quiet=True); \
    nltk.download('punkt_tab', quiet=True); \
    nltk.download('averaged_perceptron_tagger', quiet=True); \
    nltk.download('averaged_perceptron_tagger_eng', quiet=True); \
    nltk.download('maxent_ne_chunker', quiet=True); \
    nltk.download('words', quiet=True)"

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR /opt/airflow