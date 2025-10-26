FROM apache/airflow:2.10.1

USER root

# Install system dependencies for image processing + RAG web scraping
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Copy requirements files
COPY requirements-pipeline.txt /opt/airflow/requirements-pipeline.txt

# Install base pipeline dependencies (Vision + shared)
RUN pip install --no-cache-dir -r /opt/airflow/requirements-pipeline.txt

# RAG: ML/NLP Libraries
RUN pip install --no-cache-dir \
    sentence-transformers==2.2.2 \
    faiss-cpu==1.7.4 \
    tiktoken==0.5.2 \
    nltk==3.8.1

# RAG: Web Scraping
RUN pip install --no-cache-dir \
    beautifulsoup4==4.12.2 \
    playwright==1.40.0 \
    aiohttp==3.9.1 \
    newspaper3k \
    lxml \
    trafilatura \
    markdownify

# RAG: Data Validation
RUN pip install --no-cache-dir \
    tensorflow-data-validation==1.14.0 \
    great_expectations

# RAG: Install Playwright browsers for web scraping
RUN playwright install chromium

RUN pip install --no-cache-dir \
    google-cloud-storage \
    dvc \
    pyyaml

# RAG: Download NLTK data
RUN python -c "import nltk; \
    nltk.download('punkt', quiet=True); \
    nltk.download('averaged_perceptron_tagger', quiet=True); \
    nltk.download('averaged_perceptron_tagger_eng', quiet=True); \
    nltk.download('maxent_ne_chunker', quiet=True); \
    nltk.download('words', quiet=True); \
    nltk.download('maxent_treebank_pos_tagger', quiet=True)"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /opt/airflow