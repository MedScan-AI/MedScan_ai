# RAG Model Development

Retrieval-Augmented Generation (RAG) system for medical question answering, specializing in Tuberculosis (TB) and Lung Cancer information. Built with FAISS for vector search, vLLM for efficient inference, MLflow for experiment tracking, Fairlearn for bias detection, and Google Cloud Platform for deployment.

## Overview

This system combines semantic search with large language models to provide evidence-based medical information from curated literature.

### Model Selection/Experimentation Considerations

- 7 optimized LLM models (360M to 14B parameters)
- 4 retrieval strategies (similarity, MMR, weighted scoring, reranking)
- 5 carefully crafted prompt templates for different use cases
- MLflow integration for experiment tracking and model selection
- Hallucination detection using Vectara's evaluation model
- Comprehensive bias detection with Fairlearn-powered fairness analysis
- Automated bias mitigation (threshold optimization, reweighting)
- GCS/local hybrid data loading for flexible deployment
- Cloud Run deployment with endpoint management
- Comprehensive testing suite with pytest

## Project Structure
```
├── ModelInference/
│   ├── RAG_inference.py           # Main RAG pipeline with GCS support
│   ├── test_rag_pipeline.py       # Comprehensive pytest suite (400+ lines)
│   ├── deploy.py                  # Model versioning manager (registers models in Vertex AI Model Registry)
│   ├── test_deployment.py         # Pre-flight deployment checks
│   └── vertex_rag_client.py       # Client for querying deployed endpoint
│
├── ModelSelection/
│   ├── experiment.py              # HPO with Optuna + MLflow logging
│   ├── select_best_model.py       # MLflow-based model selector
│   ├── prompts.py                 # 5 prompt template variants
│   ├── rag_bias_adapter.py        # RAG-to-bias detection adapter
│   ├── rag_bias_check.py          # Comprehensive bias detector
│   └── qa.json                    # Evaluation question-answer pairs
│
├── models/
│   └── models.py                  # Unified model factory (7 LLMs)
│
├── utils/
│   ├── retreival_methods.py       # 4 retrieval strategies (note: typo in filename)
│   └── RAG_config.json            # Auto-generated best model config
│
└── data/RAG/index/
    ├── embeddings_latest.json     # Document embeddings metadata
    ├── index_latest.bin           # FAISS vector index
    └── data_latest.pkl            # Serialized document data
```

## Prerequisites

### Required Packages
```bash
# Core dependencies
pip install faiss-cpu sentence-transformers transformers
pip install vllm torch mlflow numpy pandas
pip install fairlearn scikit-learn matplotlib seaborn
pip install optuna

# Google Cloud dependencies (for deployment)
pip install google-cloud-storage google-cloud-aiplatform

# Testing dependencies
pip install pytest pytest-mock
```

### System Requirements

- **Python**: 3.10+
- **GPU**: Recommended for vLLM models (CUDA 11.8+)
- **RAM**: 16GB+ (32GB+ for larger models)
- **Disk**: 50GB+ for model weights

## Environment Setup

### Required Environment Variables
```bash
# Google Cloud Platform (required for GCS/Vertex AI)
export GCP_PROJECT_ID="your-gcp-project-id"
export GCS_BUCKET_NAME="your-gcs-bucket-name"

# MLflow (optional - defaults to local)
export MLFLOW_TRACKING_URI="http://localhost:5000"

# Data paths (optional - has smart defaults)
export EMBEDDING_PATH="/path/to/embeddings.json"
export INDEX_PATH="/path/to/index.bin"
export DATA_PATH="/path/to/data.pkl"
```

### Setup Script
```bash
# Create .env file
cat > .env << EOF
GCP_PROJECT_ID=your-project-id
GCS_BUCKET_NAME=your-bucket-name
MLFLOW_TRACKING_URI=http://localhost:5000
EOF

# Load environment
source .env

# Verify setup
python -c "import os; print('GCP Project:', os.getenv('GCP_PROJECT_ID'))"
```

## Core Components

### 1. RAG Inference Pipeline (ModelInference/RAG_inference.py)

Main orchestration file with GCS/local hybrid loading.

#### RAGDataLoader Class

```python
from ModelInference.RAG_inference import RAGDataLoader

# Initialize with GCS
loader = RAGDataLoader(
    bucket_name="my-bucket",
    project_id="my-project"
)

# Load data (tries GCS first, falls back to local)
embeddings = loader.load_embeddings()      # From GCS or local
index = loader.load_faiss_index()          # From GCS or local
data = loader.load_data_pkl()              # From GCS or local
```

#### Key Functions
```python
# Load configuration (supports GCS URIs)
config = load_config("gs://bucket/config.json")  # GCS
config = load_config("utils/RAG_config.json")    # Local

# Generate embedding
embedding = get_embedding(query, model_name="BAAI/llm-embedder")

# Retrieve documents
documents = retrieve_documents(
    embedding=embedding,
    index=faiss_index,
    embeddings_data=data,
    k=5,
    retrieval_method="similarity"  # or "mmr", "weighted_score", "rerank"
)

# Generate response
response, in_tokens, out_tokens = generate_response(
    query=query,
    documents=documents,
    config=config
)

# Compute statistics
stats = compute_stats(
    query=query,
    response=response,
    retrieved_docs=documents,
    config=config,
    prompt_template=prompt,
    in_tokens=in_tokens,
    out_tokens=out_tokens
)

# End-to-end pipeline
response, stats = run_rag_pipeline(query)
```

### 2. Model Factory (models/models.py)

Unified interface for 7 LLM models via vLLM (+ SmolLM via HuggingFace).

#### Available Models

| Model Key | HuggingFace ID | Size | Context | Notes |
|-----------|----------------|------|---------|-------|
| `qwen_2.5_1.5b` | Qwen/Qwen2.5-1.5B-Instruct | 1.5B | 128K | Efficient, multilingual |
| `smol_lm` | HuggingFaceTB/SmolLM2-360M | 360M | 8K* | Smallest model |
| `llama_3.2_3b` | meta-llama/Llama-3.2-3B-Instruct | 3B | 128K | Best overall for size |
| `qwen_2.5_7b` | Qwen/Qwen2.5-7B-Instruct | 7B | 128K | Balanced |
| `mistral_7b` | mistralai/Mistral-7B-Instruct-v0.3 | 7B | 32K | Fast inference |
| `llama_3.1_8b` | meta-llama/Meta-Llama-3.1-8B-Instruct | 8B | 128K | High quality |
| `qwen_2.5_14b` | Qwen/Qwen2.5-14B-Instruct | 14B | 128K | Best for accuracy |

*SmolLM uses HuggingFace Transformers (not vLLM), different implementation.

#### Usage
```python
from models.models import ModelFactory

# Create model instance
model = ModelFactory.create_model(
    model_key="qwen_2.5_7b",
    temperature=0.7,
    top_p=0.9,
    max_tokens=500
)

# Generate response
result = model.infer(prompt)
print(result["generated_text"])
print(f"Tokens: {result['input_tokens']} in, {result['output_tokens']} out")

# List available models
models = ModelFactory.list_models()
```

### 3. Retrieval Methods (utils/retreival_methods.py)

**Note**: Filename has typo (retreival instead of retrieval).

Four sophisticated retrieval strategies:

#### 1. Similarity Search (Default)
```python
docs = retrieve_documents(
    embedding, index, embeddings_data, k=5,
    retrieval_method="similarity"
)
```

- Fast cosine similarity search
- **Best for**: Speed-critical applications

#### 2. MMR (Maximal Marginal Relevance)
```python
docs = retrieve_documents(
    embedding, index, embeddings_data, k=5,
    retrieval_method="mmr"
)
```

- Balances relevance with diversity
- λ parameter: 0.7 (70% relevance, 30% diversity)
- **Best for**: Avoiding redundant results

#### 3. Weighted Score
```python
docs = retrieve_documents(
    embedding, index, embeddings_data, k=5,
    retrieval_method="weighted_score"
)
```

- Combines similarity with metadata signals
- Factors: recency, quality, citations
- **Best for**: When metadata quality is available

#### 4. Two-Stage Rerank
```python
docs = retrieve_documents(
    embedding, index, embeddings_data, k=5,
    retrieval_method="rerank"
)
```

- Retrieves 3× candidates, then reranks
- Uses content length and relevance heuristics
- **Best for**: Maximum precision over speed

### 4. Prompt Templates (ModelSelection/prompts.py)

Five carefully designed prompt variants:

| Prompt | Focus | Use Case |
|--------|-------|----------|
| `prompt1` | Evidence-based citations | Formal medical documentation |
| `prompt2` | Scope-limited (TB/Lung Cancer) | Focused medical QA |
| `prompt3` | Patient-centered education | Plain language explanations |
| `prompt4` | Systematic analysis | Comprehensive research |
| `prompt5` | Clinical literature | Peer-reviewed evidence synthesis |

**Key Features:**

- Source citation with metadata
- Explicit hallucination prevention
- Medical disclaimer integration
- Scope management (TB/Lung Cancer only)

**Usage:**
```python
from ModelSelection.prompts import PROMPTS

prompt = PROMPTS.version["prompt3"]  # Patient-centered
formatted = prompt.format(context=context, query=query)
```

## Model Selection & HPO

### Running Experiments (ModelSelection/experiment.py)

Hyperparameter optimization with Optuna + MLflow tracking.
```python
from ModelSelection.experiment import run_mlflow_experiment

# Define models to test
models_to_test = [
    {"name": "qwen_2.5_1.5b", "type": "open-source"},
    {"name": "llama_3.2_3b", "type": "open-source"},
    {"name": "qwen_2.5_7b", "type": "open-source"},
]

# Define search space
search_space = {
    "temperature": (0.0, 1.0),
    "top_p": (0.7, 1.0),
    "num_retrieved_docs": (3, 8),
    "retrieval_method": ["similarity", "mmr", "weighted_score", "rerank"],
    "prompt_versions": [v for _, v in PROMPTS.version.items()],
}

# Run experiment (creates MLflow runs for each model)
run_mlflow_experiment(
    experiment_name="RAG_Model_Selection",
    models_to_test=models_to_test,
    qa_dataset_path="ModelSelection/qa.json",
    search_space=search_space,
    n_trials_per_model=20
)
```

#### Evaluation Metrics:

- **Composite Score**: 0.5 × semantic_score + 0.5 × hallucination_score
- Semantic matching (keyword-based F1)
- Hallucination score (Vectara model)
- Retrieval quality
- Runtime per query (ms)
- Token counts

#### Outputs:

- MLflow runs logged to experiment
- Best config saved to `utils/RAG_config.json`
- Bias analysis results in `bias_results.json`

### Selecting Best Model (ModelSelection/select_best_model.py)

Query MLflow for best-performing model.
```python
from ModelSelection.select_best_model import ModelSelector

# Initialize selector
selector = ModelSelector(experiment_name="RAG_Model_Selection")

# Load experiment
selector.load_experiment()

# Get all runs (sorted by composite score)
runs = selector.get_all_runs()

# Select best model
best_model = selector.select_best_model()

# Display results
selector.display_results(best_model)

# Save deployment config
selector.save_best_model_config(best_model)
```

#### Generated Config Format (utils/RAG_config.json):
```json
{
  "model_name": "Qwen/Qwen2.5-7B-Instruct",
  "display_name": "qwen_2.5_7b",
  "model_type": "open-source",
  "temperature": 0.7234,
  "top_p": 0.9123,
  "k": 5,
  "retrieval_method": "mmr",
  "embedding_model": "BAAI/llm-embedder",
  "prompt": "...",
  "mlflow_run_id": "abc123...",
  "performance_metrics": {
    "composite_score": 0.8456,
    "semantic_matching_score": 0.8234,
    "hallucination_score": 0.8678,
    "retrieval_score": 0.7890
  }
}
```

**Important**: Config stores full HuggingFace IDs (e.g., `Qwen/Qwen2.5-7B-Instruct`), not short keys.

## Bias Detection & Mitigation

### Overview

Comprehensive fairness analysis using Fairlearn across multiple demographic slices.

#### Analyzed Dimensions

- **Query Complexity**: simple (<10 words), moderate (10-20), complex (>20 words)
- **Medical Domain**: tuberculosis, lung_cancer, general
- **Query Type**: symptom, diagnosis, treatment, general

#### Fairness Metrics

- Demographic parity difference
- Equalized odds difference
- Accuracy disparity across slices
- F1 score disparity
- False positive/negative rate disparities

### Bias Adapter (ModelSelection/rag_bias_adapter.py)

Converts continuous RAG scores to binary labels for fairness analysis.
```python
from ModelSelection.rag_bias_adapter import RAGBiasAdapter, run_bias_check

# Initialize adapter
adapter = RAGBiasAdapter(
    semantic_threshold=0.5,
    hallucination_threshold=0.7
)

# Run bias analysis on HPO results
bias_results = adapter.run_bias_analysis(
    per_query_results=hpo_results["per_query_results"]
)

# Or use convenience function after HPO
bias_results = run_bias_check(best_metrics)

print(f"Bias detected: {bias_results['bias_detected']}")
print(f"Violations: {bias_results['num_violations']}")
```

**Automatic Features:**

- Creates metadata from query characteristics
- Converts scores to binary labels via thresholds
- Generates slice-based analysis
- Produces mitigation recommendations

### Comprehensive Bias Detector (ModelSelection/rag_bias_check.py)

Full-featured bias detection with mitigation strategies.
```python
from ModelSelection.rag_bias_check import ComprehensiveBiasDetector

# Initialize detector
detector = ComprehensiveBiasDetector(
    model=your_model,
    bias_thresholds={
        'accuracy_disparity': 0.05,  # Max 5% accuracy gap
        'f1_disparity': 0.05,
        'demographic_parity': 0.1,
        'equalized_odds': 0.1
    }
)

# Run comprehensive analysis
result = detector.run_comprehensive_analysis(
    y_true=ground_truth_labels,
    y_pred=predicted_labels,
    metadata=query_metadata,
    slice_features=['query_complexity', 'medical_domain', 'query_type'],
    apply_mitigation=True  # Automatically apply mitigation if bias detected
)

# Generate detailed report
report = detector.generate_detailed_report(result)
print(report)

# Visualize disparities
detector.visualize_bias_analysis(result.slice_metrics)
```

### Bias Mitigation Strategies
#### 1. Threshold Optimization (Post-processing)

Uses Fairlearn's ThresholdOptimizer - NO retraining required.
```python
# Method A: Direct application
X = metadata[['y_scores']].values
mitigation_result = detector.apply_mitigation(
    X=X,
    y_true=y_true,
    sensitive_features=metadata[['query_complexity']],
    mitigation_type='threshold'
)

# Access results
y_pred_mitigated = mitigation_result['mitigated_predictions']
print(f"Mitigated accuracy: {mitigation_result['mitigated_metrics']['accuracy']:.3f}")

# Method B: Automatic (via run_comprehensive_analysis)
result = detector.run_comprehensive_analysis(
    ...,
    apply_mitigation=True  # Automatically applies ThresholdOptimizer
)
```

#### 2. Sample Reweighting (Pre-processing)

Returns sample weights for model retraining.
```python
mitigation_result = detector.apply_mitigation(
    X=X_features,
    y_true=y_true,
    sensitive_features=metadata[['query_complexity']],
    mitigation_type='reweighting'
)

# Returns weights for retraining
weights = mitigation_result['weights']
# Note: Requires model retraining with weighted loss

Available in recommendations only, not as automated strategy.

### Example: Complete Bias Workflow
```python
# 1. After HPO completes
best_metrics = {...}  # From experiment.py

# 2. Run bias check
from ModelSelection.rag_bias_adapter import run_bias_check
bias_results = run_bias_check(best_metrics)

# 3. Review results
if bias_results['bias_detected']:
    print("Bias violations found:")
    for violation in bias_results['violations']:
        print(f"  - {violation}")
    
    print("\nRecommendations:")
    for rec in bias_results['recommendations']:
        print(rec)

# 4. Apply mitigation (if needed)
detector = ComprehensiveBiasDetector(...)
result = detector.run_comprehensive_analysis(
    ...,
    apply_mitigation=True
)

# 5. Compare metrics
print(f"Original accuracy: {result.overall_metrics['accuracy']:.3f}")
if result.mitigation_applied:
    print(f"Mitigated accuracy: {result.mitigated_metrics['accuracy']:.3f}")
```

## Testing

### Unit Tests (ModelInference/test_rag_pipeline.py)

Comprehensive pytest suite (400+ lines, 40+ tests).
```bash
# Run all tests
pytest ModelInference/test_rag_pipeline.py -v

# Run specific test class
pytest ModelInference/test_rag_pipeline.py::TestLoadConfig -v

# Run with coverage
pytest ModelInference/test_rag_pipeline.py --cov=ModelInference --cov-report=html
```

#### Test Coverage:

- Configuration loading (local + GCS)
- Data loading (embeddings, FAISS, pickle)
- Embedding generation
- Document retrieval (all 4 methods)
- Response generation
- Hallucination scoring
- Statistics computation
- End-to-end pipeline
- Model factory
- Error handling

#### Key Test Classes:

- `TestLoadConfig` - Config file loading
- `TestGetEmbedding` - Embedding generation
- `TestRetrieveDocuments` - All retrieval methods
- `TestGenerateResponse` - LLM inference
- `TestComputeStats` - Statistics calculation
- `TestRunRagPipeline` - Integration tests

## Configuration

### Environment Variables
```bash
# Required for GCS/Vertex AI
export GCP_PROJECT_ID="your-gcp-project-id"
export GCS_BUCKET_NAME="your-gcs-bucket-name"

# Optional - MLflow tracking
export MLFLOW_TRACKING_URI="http://localhost:5000"

# Optional - Custom data paths (has smart defaults)
export EMBEDDING_PATH="/custom/path/embeddings.json"
export INDEX_PATH="/custom/path/index.bin"
export DATA_PATH="/custom/path/data.pkl"
```

### RAG_config.json

Auto-generated by `select_best_model.py` after HPO.
```json
{
  "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
  "display_name": "qwen_2.5_1.5b",
  "model_type": "open-source",
  "temperature": 0.8589,
  "top_p": 0.7028,
  "k": 4,
  "retrieval_method": "weighted_score",
  "embedding_model": "BAAI/llm-embedder",
  "prompt": "You are a medical information assistant that provides evidence-based responses using only the provided medical literature.\n\nCONTEXT DOCUMENTS:\n{context}\n\nINSTRUCTIONS:\n1. Answer the following question",
  "mlflow_run_id": "3b468a156fcd48f7b457ee7ce5626edc",
  "performance_metrics": {
    "composite_score": 0.2892545264635012,
    "semantic_matching_score": 0.23363830803305413,
    "hallucination_score": 0.34487074489394826,
    "retrieval_score": 0.7811772137880325,
    "api_success_rate": 100.0
  },
  "resource_requirements": {
    "memory_mb": 0.0,
    "gpu_utilization_percent": 75.0
  },
  "cost_metrics": {
    "cost_per_query_usd": 0.0,
    "runtime_per_query_ms": 3186.9593938191733
  },
  "tags": {
    "dataset_version": "QA_v1.0",
    "experiment_date": "2025-11-11",
    "is_local": "true",
    "model_family": "qwen_2.5_1.5b"
  }
}
```

## Usage Examples

### Basic Usage
```python
from ModelInference.RAG_inference import run_rag_pipeline

# Simple query
query = "What are the symptoms of tuberculosis?"
response, stats = run_rag_pipeline(query)

print(response)
print(f"\nHallucination Score: {stats['hallucination_scores']['avg']:.3f}")
print(f"Retrieval Quality: {stats['avg_retrieval_score']:.3f}")
print(f"Tokens: {stats['total_tokens']}")
```

### Custom Retrieval Strategy
```python
from ModelInference.RAG_inference import (
    load_config, load_embeddings_data, load_faiss_index,
    get_embedding, retrieve_documents
)

# Load data
config = load_config()
embeddings_data = load_embeddings_data()
index = load_faiss_index()

# Get query embedding
query = "How is lung cancer diagnosed?"
embedding = get_embedding(query)

# Try different retrieval methods
for method in ["similarity", "mmr", "weighted_score", "rerank"]:
    docs = retrieve_documents(
        embedding=embedding,
        index=index,
        embeddings_data=embeddings_data,
        k=5,
        retrieval_method=method
    )
    print(f"\n{method.upper()}:")
    for doc in docs[:2]:
        print(f"  - {doc['title']} (score: {doc['score']:.3f})")
```

### Custom Model & Prompt
```python
from ModelInference.RAG_inference import run_rag_pipeline, load_config
from ModelSelection.prompts import PROMPTS

# Modify config
config = load_config()
config['model_name'] = 'Qwen/Qwen2.5-14B-Instruct'
config['temperature'] = 0.8
config['prompt'] = PROMPTS.version['prompt3']  # Patient-centered

# Save modified config
import json
with open('utils/RAG_config.json', 'w') as f:
    json.dump(config, f, indent=2)

# Run with new config
response, stats = run_rag_pipeline(query)
```

### Running Model Selection
```python
from ModelSelection.select_best_model import ModelSelector

# Initialize selector
selector = ModelSelector(experiment_name="RAG_Model_Selection")

# Load experiment from MLflow
selector.load_experiment()

# Get all runs (sorted by composite score)
runs = selector.get_all_runs()
print(f"Found {len(runs)} runs")

# Select best model
best_model = selector.select_best_model()

# Display results
selector.display_results(best_model)

# Save deployment config
selector.save_best_model_config(best_model)
print("Config saved to utils/RAG_config.json")
```

### Evaluation on Test Set
```python
import json
from ModelInference.RAG_inference import run_rag_pipeline

# Load QA pairs
with open("ModelSelection/qa.json") as f:
    qa_pairs = json.load(f)

results = []
for item in qa_pairs:
    response, stats = run_rag_pipeline(item["Q"])
    
    results.append({
        "question": item["Q"],
        "reference": item["A"],
        "generated": response,
        "stats": stats
    })

# Calculate metrics
avg_hallucination = sum(r['stats']['hallucination_scores']['avg'] for r in results) / len(results)
print(f"Average Hallucination Score: {avg_hallucination:.3f}")
```

### Deployment Example
```bash
# Full deployment workflow

# 1. Pre-flight checks
python ModelInference/test_deployment.py

# 2. Deploy model
python ModelInference/deploy.py \
  --config utils/RAG_config.json \
  --index gs://my-bucket/RAG/index/index_latest.bin \
  --embeddings gs://my-bucket/RAG/index/embeddings_latest.json \
  --metadata deployment_metadata.json

# 3. Query endpoint
python ModelInference/vertex_rag_client.py
```

### Development Workflow

#### Complete Pipeline
```bash
# 1. Set up environment
export GCP_PROJECT_ID="your-project"
export GCS_BUCKET_NAME="your-bucket"

# 2. Run data pipeline (if needed)
# This creates index_latest.bin, embeddings_latest.json in GCS

# 3. Run model selection with HPO
python ModelSelection/experiment.py

# 4. Select best model
python ModelSelection/select_best_model.py

# 5. Run tests
pytest ModelInference/test_rag_pipeline.py -v

# 6. Pre-flight checks
python ModelInference/test_deployment.py

# 7. Deploy to Vertex AI
python ModelInference/deploy.py \
  --config utils/RAG_config.json \
  --index gs://${GCS_BUCKET_NAME}/RAG/index/index_latest.bin \
  --embeddings gs://${GCS_BUCKET_NAME}/RAG/index/embeddings_latest.json \
  --metadata metadata.json

# 8. Query deployed endpoint
python ModelInference/vertex_rag_client.py
```

## API Reference

### Core Functions

`run_rag_pipeline(query: str) -> Tuple[str, Dict]`

End-to-end RAG pipeline.

**Args:**
- `query (str)`: User question

**Returns:**
- `response (str)`: Generated answer with references and disclaimer
- `stats (dict)`: Performance statistics

**Example:**
```python
response, stats = run_rag_pipeline("What is tuberculosis?")
```

`retrieve_documents(embedding, index, embeddings_data, k, retrieval_method) -> List[Dict]`

Retrieve documents from FAISS index.

**Args:**
- `embedding (np.ndarray)`: Query embedding vector
- `index (faiss.Index)`: FAISS index
- `embeddings_data (List[Dict])`: Document metadata
- `k (int)`: Number of documents to retrieve
- `retrieval_method (str)`: "similarity", "mmr", "weighted_score", or "rerank"

**Returns:**
- List of document dictionaries with content and scores

`generate_response(query, documents, config) -> Tuple[str, int, int]`

Generate LLM response.

**Args:**
- `query (str)`: User question
- `documents (List[Dict])`: Retrieved documents
- `config (Dict)`: Configuration dictionary

**Returns:**
- `response (str)`: Generated text
- `input_tokens (int)`: Number of input tokens
- `output_tokens (int)`: Number of output tokens

### Model Factory

`ModelFactory.create_model(model_key, temperature, top_p, max_tokens) -> Model`

Create model instance.

**Args:**
- `model_key (str)`: Model identifier (see Available Models table)
- `temperature (float)`: Sampling temperature (0.0-1.0)
- `top_p (float)`: Top-p sampling (0.0-1.0)
- `max_tokens (int)`: Maximum output tokens

**Returns:**
- Model instance with `infer()` method

### Bias Detection

`ComprehensiveBiasDetector.run_comprehensive_analysis(y_true, y_pred, metadata, slice_features, apply_mitigation) -> BiasAnalysisResult`

Run complete bias analysis.

**Args:**
- `y_true (np.ndarray)`: Ground truth labels
- `y_pred (np.ndarray)`: Predicted labels
- `metadata (pd.DataFrame)`: Query metadata
- `slice_features (List[str])`: Features to analyze
- `apply_mitigation (bool)`: Auto-apply mitigation if bias detected

**Returns:**
- `BiasAnalysisResult` object with metrics, violations, recommendations

# Feature analysis

## Hyperparameter sensitivity
We integrate this into model selection by exploring the hyperparameter space and choosing the variant that achieves the strongest composite evaluation score.

## Response generation behavior
The model produces outputs solely from the provided text chunks. No auxiliary parameters or metadata influence its generation process.

## Retrieval focus insights
The ELK stack in the upcoming assignment will allow us to examine how retrieval mechanisms emphasize specific regions within the embedding space. This setup mirrors real world deployment conditions and offers a more reliable view of system behavior.
