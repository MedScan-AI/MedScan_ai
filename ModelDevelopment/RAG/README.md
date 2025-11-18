# RAG Model Development

Model selection and end to end answering and evaluation pipeline for Retrieval-Augmented Generation (RAG) system for medical question answering, specializing in Tuberculosis (TB) and Lung Cancer information. Built with FAISS for vector search, vLLMs for efficient inference, MLflow for experiment tracking, and Fairlearn for comprehensive bias detection.

## Overview

This system combines semantic search with large language models to provide evidence-based medical information from curated literature. It features:

- **Multiple retrieval strategies** (similarity, MMR, weighted scoring, reranking)
- **7 optimized LLM models** (from 360M to 14B parameters)
- **5 carefully crafted prompt templates** for different use cases
- **MLflow integration** for experiment tracking and model selection
- **Hallucination detection** using Vectara's evaluation model
- **Comprehensive bias detection** with Fairlearn-powered fairness analysis
- **Automated bias mitigation** (threshold optimization, reweighting, resampling)
- **Comprehensive metrics** (retrieval quality, hallucination scores, fairness violations, performance stats)

## Project Structure

```
├── ModelInference/
│   └── RAG_inference.py          # Main RAG pipeline orchestration
├── ModelSelection/
│   ├── prompts.py                # 5 prompt template variants
│   ├── select_best_model.py     # MLflow-based model selector
│   ├── rag_bias_adapter.py      # RAG-to-bias detection adapter
│   ├── rag_bias_check.py        # Comprehensive bias detection system
│   └── qa.json                   # Evaluation question-answer pairs
├── models/
│   └── models.py                 # Unified model factory (7 LLMs)
├── utils/
│   ├── retrieval_methods.py     # 4 retrieval strategies
│   └── RAG_config.json          # Auto-generated best model config
└── data/RAG/index/
    ├── embeddings.json           # Document embeddings metadata
    ├── index.bin                 # FAISS vector index
    └── data.pkl                  # Serialized document data
```

## Quick Start

### Prerequisites

```bash
pip install faiss-cpu sentence-transformers transformers
pip install vllm torch mlflow numpy
pip install fairlearn scikit-learn pandas matplotlib seaborn
```

### Basic Usage

```python
from RAG_inference import run_rag_pipeline

# Ask a medical question
query = "What are the symptoms of tuberculosis?"
response, stats = run_rag_pipeline(query)

print(response)
print(f"Hallucination Score: {stats['hallucination_scores']['avg']}")
print(f"Retrieval Quality: {stats['avg_retrieval_score']}")
```

### Running Model Selection

```python
from select_best_model import ModelSelector

selector = ModelSelector(experiment_name="RAG_Model_Selection")
selector.load_experiment()
selector.get_all_runs()
best_model = selector.select_best_model()
selector.display_results(best_model)
selector.save_best_model_config(best_model)
```

## Core Components

### 1. RAG Inference Pipeline (`RAG_inference.py`)

Main orchestration file that coordinates:
- **Configuration loading** from `RAG_config.json`
- **Embedding generation** using sentence transformers
- **Document retrieval** from FAISS index
- **Response generation** with selected LLM
- **Hallucination evaluation** with Vectara model
- **Statistics computation** (tokens, scores, metadata)

**Key Functions:**
```python
load_config()              # Load RAG configuration
get_embedding(query)       # Generate query embedding
retrieve_documents()       # Fetch relevant docs from FAISS
generate_response()        # LLM-based answer generation
compute_stats()           # Calculate performance metrics
run_rag_pipeline()        # End-to-end orchestration
```

### 2. Model Factory (`models/models.py`)

Unified interface for 7 LLM models via vLLM:

| Model | Size | Context
|-------|------|---------|
| `qwen_2.5_1.5b` | 1.5B | 128K
| `smol_lm` | 360M | 8K
| `llama_3.2_3b` | 3B | 128K 
| `qwen_2.5_7b` | 7B | 128K
| `mistral_7b` | 7B | 32K
| `llama_3.1_8b` | 8B | 128K
| `qwen_2.5_14b` | 14B | 128K

**Usage:**
```python
from models import ModelFactory

model = ModelFactory.create_model(
    "qwen_2.5_7b",
    temperature=0.7,
    top_p=0.9,
    max_tokens=500
)

result = model.infer(prompt)
print(result["generated_text"])
```

### 3. Retrieval Methods (`utils/retrieval_methods.py`)

Four sophisticated retrieval strategies:

#### **Similarity Search** (Default)
- Fast cosine similarity search
- Best for: Speed-critical applications

#### **MMR (Maximal Marginal Relevance)**
- Balances relevance with diversity
- Configurable λ parameter (default: 0.7)
- Best for: Avoiding redundant results

#### **Weighted Score**
- Combines similarity with metadata signals
- Factors: recency, quality, citations
- Best for: When metadata quality is available

#### **Two-Stage Rerank**
- Retrieves 3× candidates, then reranks
- Uses content length and relevance heuristics
- Best for: Maximum precision over speed

**Configuration:**
```json
{
  "retrieval_method": "mmr",
  "k": 5
}
```

### 4. Prompt Templates (`ModelSelection/prompts.py`)

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

### 5. Model Selection (`ModelSelection/select_best_model.py`)

MLflow-powered automatic model selection:

**Selection Criteria:**
```
composite_score = 0.5 × semantic_score + 0.5 × hallucination_score
```

**Tracked Metrics:**
- Semantic matching quality
- Hallucination rates
- Retrieval effectiveness
- Inference speed (ms/query)
- Cost per query (USD)
- Memory usage (MB)
- GPU utilization (%)

**Output:**
- Generates `utils/RAG_config.json` with best model configuration
- Ready for production deployment

### 6. Bias Detection System

#### A. Bias Adapter (`ModelSelection/rag_bias_adapter.py`)

Converts continuous RAG scores to binary labels for fairness analysis:

**Key Features:**
- **Automatic metadata creation** from query characteristics
- **Score thresholding** (semantic + hallucination → binary labels)
- **MLflow integration** for bias tracking
- **Query categorization** by complexity, domain, and type

**Usage:**
```python
from rag_bias_adapter import RAGBiasAdapter, run_bias_check

# After HPO/evaluation completes
bias_results = run_bias_check(best_metrics)

# Or use adapter directly
adapter = RAGBiasAdapter(
    semantic_threshold=0.5,
    hallucination_threshold=0.7
)

bias_analysis = adapter.run_bias_analysis(per_query_results)
adapter.log_to_mlflow(bias_analysis)
```

**Query Slicing Dimensions:**
- **Query Complexity**: simple (<10 words), moderate (10-20), complex (>20 words)
- **Medical Domain**: tuberculosis, lung_cancer, general
- **Query Type**: symptom, diagnosis, treatment, general

#### B. Comprehensive Bias Detector (`ModelSelection/rag_bias_check.py`)

Full-featured bias detection and mitigation using Fairlearn:

**Fairness Metrics:**
- Demographic parity difference
- Equalized odds difference
- Accuracy disparity across slices
- F1 score disparity
- False positive/negative rate disparities

**Analysis Features:**
- **Slice-based analysis** across multiple dimensions
- **Disparity detection** with configurable thresholds
- **Violation reporting** with detailed explanations
- **Automated recommendations** for mitigation

**Mitigation Strategies:**

1. **Threshold Optimization** (Post-processing)
   - Adjust decision thresholds per demographic group
   - Uses Fairlearn's `ThresholdOptimizer`
   - No model retraining required

2. **Sample Reweighting** (Pre-processing)
   - Inverse frequency weighting for underrepresented groups
   - Requires model retraining with weighted loss

3. **Resampling** (Pre-processing)
   - Oversample underrepresented slices
   - Generate synthetic data for minority groups

**Usage:**
```python
from rag_bias_check import ComprehensiveBiasDetector

# Initialize detector
detector = ComprehensiveBiasDetector(
    model=your_model,
    bias_thresholds={
        'accuracy_disparity': 0.05,  # 5% max accuracy difference
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
    apply_mitigation=True
)

# Generate report
report = detector.generate_detailed_report(result)
print(report)

# Visualize disparities
detector.visualize_bias_analysis(result.slice_metrics)
```

**Output Structure:**
```python
BiasAnalysisResult(
    # Overall performance metrics
    overall_metrics={
        'accuracy': 0.85,
        'precision': 0.82,
        'recall': 0.87,
        'f1_score': 0.84
    },
    
    # Per-slice detailed metrics
    slice_metrics=[
        SliceMetrics(
            slice_name='query_complexity',
            slice_value='complex',
            sample_size=50,
            accuracy=0.78,
            precision=0.75,
            recall=0.80,
            f1_score=0.75,
            false_positive_rate=0.15, 
            false_negative_rate=0.12
        ),
        SliceMetrics(
            slice_name='query_complexity',
            slice_value='simple',
            sample_size=100,
            accuracy=0.93,
            precision=0.91,
            recall=0.94,
            f1_score=0.92,
            false_positive_rate=0.05,
            false_negative_rate=0.08
        ),
        # ... more slices
    ],
    
    # Disparity analysis
    disparities={
        'query_complexity': {
            'accuracy_range': 0.15,
            'accuracy_std': 0.06,
            'f1_range': 0.17, 
            'f1_std': 0.07,
            'fpr_range': 0.10,
            'fnr_range': 0.04,
            'worst_performing_slice': 'complex',
            'best_performing_slice': 'simple'
        }
    },
    
    # Fairness violations (if any)
    fairness_violations=[
        "query_complexity: Accuracy disparity (0.15) exceeds threshold (0.05)",
        "query_complexity: F1 score disparity (0.17) exceeds threshold (0.05)"
    ],
    
    # Whether bias was detected
    bias_detected=True, 
    # Mitigation recommendations
    mitigation_recommendations=[      
        "1. Re-sampling Strategy for query_complexity:\n"
        "   - Oversample data from 'complex' group (current worst performer)\n"
        "   - Consider synthetic data generation for underrepresented cases",
        "2. Re-weighting Strategy for query_complexity:\n"
        "   - Apply higher weights to 'complex' samples during training\n"
        "   - Suggested weight ratio: 1.15",
        "3. Threshold Optimization for query_complexity:\n"
        "   - Adjust decision thresholds per group to equalize performance\n"
        "   - Consider using ThresholdOptimizer from Fairlearn"
    ],
    
    # Mitigation status (if applied)
    mitigation_applied=False,
    mitigated_metrics=None
)
```

**Visualization Output:**
- **Accuracy by slice** with worst/best performers highlighted
- **F1 score comparison** across demographic groups
- **Sample size annotations** for statistical significance
- **Disparity heatmaps** showing fairness violations

## Configuration

### RAG_config.json

```json
{
  "model_name": "qwen_2.5_7b",
  "model_type": "vllm",
  "temperature": 0.7,
  "top_p": 0.9,
  "k": 5,
  "retrieval_method": "similarity",
  "embedding_model": "BAAI/llm-embedder",
  "prompt": "<your_selected_prompt_template>",
  "mlflow_run_id": "abc123..."
}
```

### Environment Variables

```bash
# Optional: MLflow tracking
export MLFLOW_TRACKING_URI="http://localhost:5000"

# Data paths (default: /opt/airflow/DataPipeline/data/RAG/index/)
export EMBEDDING_PATH="/path/to/embeddings.json"
export INDEX_PATH="/path/to/index.bin"
export DATA_PATH="/path/to/data.pkl"
```

## Evaluation & Metrics

### Automatic Metrics

**Retrieval Quality:**
- Average similarity score across top-k documents
- Per-document relevance scoring

**Hallucination Detection:**
- Vectara's hallucination evaluation model
- Per-source hallucination scores
- Aggregated average across all sources

**Performance Stats:**
- Input/output token counts
- Total tokens per query
- Runtime metrics
- Memory usage

**Fairness & Bias Metrics:**
- Demographic parity difference
- Equalized odds difference
- Accuracy/F1 disparity across slices
- False positive/negative rate disparities
- Per-slice performance analysis

### Running Experiments

```python
# Run model selection experiments
python ModelSelection/select_best_model.py

# Evaluate on test set
with open("ModelSelection/qa.json") as f:
    qa_pairs = json.load(f)

for item in qa_pairs:
    response, stats = run_rag_pipeline(item["question"])
    # Compare with item["answer"]
```

### Running Bias Analysis

```python
from rag_bias_adapter import run_bias_check

# After hyperparameter optimization
best_metrics = {
    "per_query_results": [
        {
            "query": "What are symptoms of TB?",
            "semantic_matching_score": 0.85,
            "hallucination_score": 0.92
        },
        # ... more results
    ]
}

# Run comprehensive bias check
bias_results = run_bias_check(best_metrics)

# Results logged to MLflow automatically
# - bias_detected: bool
# - bias_violations_count: int
# - bias_overall_accuracy: float
# - violations.json: detailed violations
# - recommendations.json: mitigation strategies
```

## Features

### Custom Retrieval Strategy

```python
from retrieval_methods import retrieve_documents

documents = retrieve_documents(
    embedding=query_embedding,
    index=faiss_index,
    embeddings_data=data,
    k=10,
    retrieval_method="mmr"  # or "weighted_score", "rerank"
)
```

### Custom Prompt Engineering

```python
from prompts import PROMPTS

custom_prompt = PROMPTS.version["prompt3"]  # Patient-centered
config["prompt"] = custom_prompt
```

### Hallucination Mitigation

The system uses multiple strategies:
1. **Context-only responses** (no external knowledge)
2. **Explicit source citations** (document references)
3. **Automatic evaluation** (Vectara model scoring)
4. **Medical disclaimers** (safety notices)

### Bias Mitigation

The system implements bias mitigation through the `apply_mitigation()` method:

#### 1. Pre-processing Mitigation (Sample Reweighting)
```python
from rag_bias_check import ComprehensiveBiasDetector

# Initialize detector
detector = ComprehensiveBiasDetector(model=None)

# Calculate sample weights for reweighting
weights = detector._calculate_sample_weights(y_true, sensitive_features)

# The apply_mitigation method with 'reweighting' type returns these weights
mitigation_result = detector.apply_mitigation(
    X=X_features,
    y_true=y_true,
    sensitive_features=metadata[['query_complexity']],
    mitigation_type='reweighting'
)

# Returns: {'recommendation': 'Retrain model with weighted samples', 'weights': weights}
# Note: Requires model retraining with the calculated weights
```

#### 2. Post-processing Mitigation (Threshold Optimization)
```python
# Apply threshold optimization (no retraining needed)
from rag_bias_check import ComprehensiveBiasDetector

detector = ComprehensiveBiasDetector(
    model=your_model,
    bias_thresholds={'accuracy_disparity': 0.05, 'f1_disparity': 0.05}
)

# Option A: Use apply_mitigation directly
X = metadata[['y_scores']].values  # Feature matrix
mitigation_result = detector.apply_mitigation(
    X=X,
    y_true=y_true,
    sensitive_features=metadata[['query_complexity']],
    mitigation_type='threshold'  # Uses Fairlearn's ThresholdOptimizer
)

# Access results
y_pred_mitigated = mitigation_result['mitigated_predictions']
print(f"Mitigated accuracy: {mitigation_result['mitigated_metrics']['accuracy']:.3f}")
print(f"Demographic parity: {mitigation_result['mitigated_metrics']['demographic_parity_difference']:.3f}")

# Option B: Let run_comprehensive_analysis handle it automatically
result = detector.run_comprehensive_analysis(
    y_true=y_true,
    y_pred=y_pred,
    metadata=metadata,
    slice_features=['query_complexity', 'medical_domain'],
    apply_mitigation=True  # Automatically applies ThresholdOptimizer if bias detected
)

# Check if mitigation was applied
if result.mitigation_applied:
    print(f"Original accuracy: {result.overall_metrics['accuracy']:.3f}")
    print(f"Mitigated accuracy: {result.mitigated_metrics['accuracy']:.3f}")
    print(f"Demographic parity: {result.mitigated_metrics['demographic_parity_difference']:.3f}")
```

**Note:** The system currently implements two mitigation strategies:
- **Reweighting** (`mitigation_type='reweighting'`): Returns sample weights for model retraining
- **Threshold Optimization** (`mitigation_type='threshold'`): Applies Fairlearn's `ThresholdOptimizer` for post-processing fairness

### Custom Bias Thresholds

```python
# Configure custom fairness thresholds
detector = ComprehensiveBiasDetector(
    model=your_model,
    bias_thresholds={
        'demographic_parity': 0.05,     # Max 5% selection rate difference
        'equalized_odds': 0.08,         # Max 8% TPR/FPR difference
        'accuracy_disparity': 0.03,     # Max 3% accuracy gap
        'f1_disparity': 0.04            # Max 4% F1 score gap
    }
)
```
