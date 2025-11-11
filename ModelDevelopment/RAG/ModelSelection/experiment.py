import logging
import json
import time
import re
import os
import sys
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import mlflow
import mlflow.sklearn
import optuna
from optuna.samplers import TPESampler
import faiss
from datetime import datetime
from retreival_methods import retrieve_documents
from prompts import PROMPTS
from models import ModelFactory
from vllm import SamplingParams

# Reuse existing imports from your code
cur_dir = os.path.dirname(__file__) # ModelSelection/
path = os.path.dirname(cur_dir) # RAG/
sys.path.insert(0, path)
from ModelInference.RAG_inference import (
    load_embeddings_data,
    load_faiss_index,
    get_embedding,
    compute_hallucination_score
)
from transformers import AutoModelForSequenceClassification
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clear_hf_model_cache(model_name: str):
    """
    Delete cached files for a specific Hugging Face model.
    Works for all huggingface-hub versions.
    """

    # Clear GPU memory 
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Force reset of memory allocator
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
    
    cache_dir = os.path.expanduser("/projects/aiwell/conda/envs/ToM/.cache/hub/")
    prefix = "models--"

    if not os.path.exists(cache_dir):
        logging.warning(f"Cache directory does not exist: {cache_dir}")
        return

    deleted = False
    for folder in os.listdir(cache_dir):
        if folder.startswith(prefix):
            path = os.path.join(cache_dir, folder)
            try:
                shutil.rmtree(path)
                logging.info(f"Deleted cache: {path}")
                deleted = True
            except Exception as e:
                logging.error(f"Failed to delete {path}: {e}")

    if not deleted:
        logging.warning("No model caches found to delete.")


def generate_response(
    query: str,
    documents: List[Dict[str, Any]],
    config: Dict[str, Any],
    model: Any
) -> Optional[Tuple[str, int, int]]:
    """
    Generate response using LLM with retrieved context.
    
    Args:
        query: Original user query
        documents: Retrieved documents with metadata
        config: Configuration containing model params and prompt
        
    Returns:
        Tuple of (response string, input_tokens, output_tokens) or None on failure
    """
    try:
        logger.info("Generating response")
        
        # Extract config parameters
        model_name = config.get("model_name", None)
        model_type = config.get("model_type", None)
        prompt_template = config.get("prompt", None)
        
        if model_name == None or model_type == None or prompt_template == None:
            logger.error("Model details/ Prompt cannot be empty")
            sys.exit(1)
        
        temperature = config.get("temperature", 0.7)
        top_p = config.get("top_p", 0.9)
        
        # Format context from documents with actual content
        context_parts = []
        for d in documents:
            title = d.get('title', 'Unknown')
            content = d.get('content', '')
            context_parts.append(f"Document {d['rank']} - {title}:\n{content}")
        
        context = "\n\n".join(context_parts)
        
        # Format prompt
        formatted_prompt = prompt_template.format(context=context, query=query)
        
        logger.info(f"Using model: {model_name} (type: {model_type})")
        logger.info(f"Temperature: {temperature}, Top-p: {top_p}")
        
        response_d = model.infer(formatted_prompt)

        if response_d is None or response_d.get('success') != True:
            raise Exception(f"Error generating response - {response_d}")  
        
        response = response_d.get('generated_text', None)
        in_tokens = response_d.get('input_tokens', 0)
        out_tokens = response_d.get('output_tokens', 0)

        
        # Add references section with links
        references = "\n\n**References:**\n"
        for d in documents:
            link = d.get('metadata', {}).get('link', '')
            title = d.get('title', 'Unknown')
            if link:
                references += f"{d['rank']}. [{title}]({link})\n"
            else:
                references += f"{d['rank']}. {title}\n"
        
        response += references

        logger.info("Response generated successfully")
        return response, in_tokens, out_tokens
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return None

def compute_answer_relevance(
    generated_answer: str,
    reference_answer: str
) -> Dict[str, float]:
    """
    Compute keyword-based relevance metrics between generated and reference answers
    
    This uses lexical matching to measure overlap at word and phrase level.
    Complements semantic similarity with explicit keyword matching.
    
    Args:
        generated_answer: The answer generated by the model
        reference_answer: The ground truth reference answer
        
    Returns:
        Dict[str, float]: Dictionary containing:
            - keyword_precision: How many keywords in generated are in reference (0-1)
            - keyword_recall: How many keywords from reference are in generated (0-1)
            - keyword_f1: Harmonic mean of precision and recall (0-1)
            - unigram_overlap: Jaccard similarity of word sets (0-1)
            - bigram_overlap: Jaccard similarity of consecutive word pairs (0-1)
            
            Higher values are better for all metrics.
    """
    
    def preprocess(text: str) -> List[str]:
        """
        Preprocess text: lowercase, remove punctuation, filter stop words
        
        Args:
            text: Input text
            
        Returns:
            List of cleaned words
        """
        # Lowercase
        text = text.lower()
        
        # Remove punctuation (keep only alphanumeric and spaces)
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Split into words
        words = text.split()
        
        # Define stop words (common words to ignore)
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'of', 'with', 'by', 'from', 'is', 'was',
            'are', 'were', 'been', 'be', 'have', 'has', 'had', 'do',
            'does', 'did', 'will', 'would', 'could', 'should', 'may',
            'might', 'can', 'this', 'that', 'these', 'those', 'it'
        }
        
        # Filter: remove stop words and words shorter than 3 characters
        filtered = [w for w in words if w not in stop_words and len(w) > 2]
        
        return filtered
    
    def get_bigrams(words: List[str]) -> List[Tuple[str, str]]:
        """
        Extract bigrams (consecutive word pairs) from word list
        
        Args:
            words: List of words
            
        Returns:
            List of bigram tuples
        """
        return [(words[i], words[i+1]) for i in range(len(words)-1)]
    
    # Preprocess both answers
    gen_words = preprocess(generated_answer)
    ref_words = preprocess(reference_answer)
    
    # Handle empty texts
    if not gen_words or not ref_words:
        return {
            "keyword_precision": 0.0,
            "keyword_recall": 0.0,
            "keyword_f1": 0.0,
            "unigram_overlap": 0.0,
            "bigram_overlap": 0.0
        }
    
    # Convert to sets for comparison
    gen_set = set(gen_words)
    ref_set = set(ref_words)
    
    # Compute unigram metrics
    common_words = gen_set.intersection(ref_set)
    
    # Precision: How many of generated words are in reference
    precision = len(common_words) / len(gen_set) if gen_set else 0.0
    
    # Recall: How many of reference words are in generated
    recall = len(common_words) / len(ref_set) if ref_set else 0.0
    
    # F1: Harmonic mean of precision and recall
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    
    # Jaccard similarity for unigrams
    union = gen_set.union(ref_set)
    unigram_jaccard = len(common_words) / len(union) if union else 0.0
    
    # Compute bigram overlap
    gen_bigrams = set(get_bigrams(gen_words))
    ref_bigrams = set(get_bigrams(ref_words))
    
    if gen_bigrams and ref_bigrams:
        common_bigrams = gen_bigrams.intersection(ref_bigrams)
        bigram_union = gen_bigrams.union(ref_bigrams)
        bigram_jaccard = (
            len(common_bigrams) / len(bigram_union) 
            if bigram_union else 0.0
        )
    else:
        bigram_jaccard = 0.0
    
    return {
        "keyword_precision": precision,
        "keyword_recall": recall,
        "keyword_f1": f1,
        "unigram_overlap": unigram_jaccard,
        "bigram_overlap": bigram_jaccard
    }

def load_qa_dataset(qa_path: str) -> List[Dict[str, str]]:
    """
    Load QA dataset from JSON file.
    
    Args:
        qa_path: Path to QA JSON file
        
    Returns:
        List of QA pairs, each containing 'question' and 'answer' keys
    """
    try:
        logger.info(f"Loading QA dataset from {qa_path}")
        with open(qa_path, 'r') as f:
            qa_data = json.load(f)
        logger.info(f"Loaded {len(qa_data)} QA pairs")
        return qa_data
    except Exception as e:
        logger.error(f"Error loading QA dataset: {e}")
        raise

def evaluate_single_query(
    qa_pair: Dict[str, str],
    config: Dict[str, Any],
    embeddings_data: List[Dict[str, Any]],
    index: Any,
    hallucination_model: Any,
    model: Any
) -> Dict[str, Any]:
    """
    Evaluate RAG pipeline on a single query and compute all metrics.
    
    Args:
        qa_pair: Dictionary with 'question' and 'answer' keys
        config: Configuration dict with model params
        embeddings_data: Loaded embeddings data
        index: FAISS index
        hallucination_model: Pre-loaded hallucination evaluation model
        
    Returns:
        Dictionary containing all metrics for this query
    """
    try:
        query = qa_pair["Q"]
        reference_answer = qa_pair["A"]
        
        start_time = time.time()
        
        # 1. Get embedding
        embedding = get_embedding(query, "BAAI/llm-embedder")
        if embedding is None:
            raise Exception("Failed to generate embedding")
        
        # 2. Retrieve documents
        k = config.get("num_retrieved_docs", 5)
        retrieval_method = config.get("retrieval_method", "similarity")
        documents = retrieve_documents(embedding, index, embeddings_data, k, retrieval_method)
        if not documents:
            raise Exception("Failed to retrieve documents")
        
        # 3. Generate response
        result = generate_response(query, documents, config, model)
        if result is None:
            raise Exception("Failed to generate response")
        
        response, in_tokens, out_tokens = result
        
        # Calculate runtime
        end_time = time.time()
        runtime_ms = (end_time - start_time) * 1000
        
        # 4. Compute semantic matching score (keyword-based)
        relevance_metrics = compute_answer_relevance(response, reference_answer)
        # Use F1 as primary semantic matching score
        semantic_matching_score = relevance_metrics["keyword_f1"]
        
        # 5. Compute hallucination score
        hallucination_scores = []
        context_texts = [doc.get("content", "") for doc in documents]
        
        pairs = [(response, context) for context in context_texts if context]
        if pairs:
            scores = compute_hallucination_score(pairs, context_texts[0], hallucination_model)
            if scores is not None:
                if isinstance(scores, list):
                    hallucination_scores = scores
                else:
                    hallucination_scores = [scores]
        
        avg_hallucination_score = (
            sum(hallucination_scores) / len(hallucination_scores) 
            if hallucination_scores else 0.0
        )
        
        # 6. Compute retrieval score (avg similarity score)
        retrieval_scores = [doc.get("score", 0.0) for doc in documents]
        avg_retrieval_score = sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0.0
        
        # 7. Compute cost per query
        model_type = config.get("model_type", "openai")
        cost_per_query = 0.0
        if model_type == "openai":
            # Rough pricing (adjust based on actual model)
            cost_per_query = (in_tokens * 0.00001 + out_tokens * 0.00003)
        elif model_type == "anthropic":
            cost_per_query = (in_tokens * 0.00003 + out_tokens * 0.00015)
        # For open-source models, cost is 0
        
        # 8. API success rate
        api_success_rate = 100.0  # Success if we got here
        
        # 9. Memory and GPU usage (placeholders - need actual implementation)
        memory_usage_mb = 0.0
        gpu_utilization_percent = 0.0
        
        if model_type in ["local", "open-source"]:
            # Estimate model size (simplified)
            model_name = config.get("model_name", "")
            if "7b" in model_name.lower():
                memory_usage_mb = 14000  # ~14GB for 7B model
            elif "13b" in model_name.lower():
                memory_usage_mb = 26000
            elif "70b" in model_name.lower():
                memory_usage_mb = 140000
            
            # GPU utilization - would need actual monitoring
            gpu_utilization_percent = 75.0  # Placeholder
        
        # After generating response
        logger.info(f"Generated response length: {len(response) if response else 0}")
        logger.info(f"First 200 chars: {response[:200] if response else 'NONE'}")
        logger.info(f"Semantic score: {semantic_matching_score}")
        logger.info(f"Hallucination score: {avg_hallucination_score}")

        return {
            "query": query,
            "reference_answer": reference_answer,
            "generated_answer": response,
            "semantic_matching_score": semantic_matching_score,
            "hallucination_score": avg_hallucination_score,
            "retrieval_score": avg_retrieval_score,
            "runtime_per_query_ms": runtime_ms,
            "cost_per_query_usd": cost_per_query,
            "api_success_rate": api_success_rate,
            "memory_usage_mb": memory_usage_mb,
            "gpu_utilization_percent": gpu_utilization_percent,
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "relevance_details": relevance_metrics,
            "num_retrieved_docs": len(documents)
        }
        
    except Exception as e:
        logger.error(f"Error evaluating query '{qa_pair.get('Q', 'N/A')}': {e}")
        return {
            "query": qa_pair.get("Q", ""),
            "error": str(e),
            "semantic_matching_score": 0.0,
            "hallucination_score": 0.0,
            "retrieval_score": 0.0,
            "runtime_per_query_ms": 0.0,
            "cost_per_query_usd": 0.0,
            "api_success_rate": 0.0,
            "memory_usage_mb": 0.0,
            "gpu_utilization_percent": 0.0
        }

def evaluate_on_qa_dataset(
    qa_dataset: List[Dict[str, str]],
    config: Dict[str, Any],
    embeddings_data: List[Dict[str, Any]],
    model,
    hallucination_model,
    index: Any
) -> Dict[str, Any]:
    """
    Evaluate RAG pipeline on entire QA dataset and aggregate metrics.
    """
    logger.info(f"Evaluating on {len(qa_dataset)} QA pairs")
    
    # # Extract config parameters
    # model_name = config.get("model_name", None)
    # if model_name == None:
    #     logger.error("Incorrect Model Name")
    #     sys.exit(1)    
    # temperature = config.get("temperature", 0.7)
    # top_p = config.get("top_p", 0.9)

    # try:    
    #     model = ModelFactory.create_model(model_name, temperature, top_p)
    # except Exception as e:
    #     logger.error(f"FAILED to load model: {e}")  # Change to error
    #     logger.exception(e)  # ADD THIS to see full traceback
    #     return {
    #         "avg_semantic_matching_score": 0.0,
    #         "avg_hallucination_score": 0.0,
    #         "avg_retrieval_score": 0.0,
    #         "avg_runtime_per_query_ms": 0.0,
    #         "avg_cost_per_query_usd": 0.0,
    #         "api_success_rate": 0.0,
    #         "avg_memory_usage_mb": 0.0,
    #         "avg_gpu_utilization_percent": 0.0,
    #         "total_queries": len(qa_dataset),
    #         "failed_queries": len(qa_dataset),
    #         "per_query_results": []
    #     }
    
    results = []
    for qa_pair in qa_dataset:
        result = evaluate_single_query(
            qa_pair, 
            config, 
            embeddings_data, 
            index, 
            hallucination_model,
            model
        )
        results.append(result)
    
    # Clean up models before aggregating results**
    try:
        if model is not None:
            del model
        if hallucination_model is not None:
            del hallucination_model
        
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        logger.info(f"Cleaned up model")
    except Exception as e:
        logger.warning(f"Error during cleanup: {e}")
    
    # Aggregate metrics
    successful_results = [r for r in results if "error" not in list(r.keys())]
    failed_count = len(results) - len(successful_results)
    
    if not successful_results:
        logger.error("All queries failed!")
        return {
            "avg_semantic_matching_score": 0.0,
            "avg_hallucination_score": 0.0,
            "avg_retrieval_score": 0.0,
            "avg_runtime_per_query_ms": 0.0,
            "avg_cost_per_query_usd": 0.0,
            "api_success_rate": 0.0,
            "avg_memory_usage_mb": 0.0,
            "avg_gpu_utilization_percent": 0.0,
            "total_queries": len(qa_dataset),
            "failed_queries": failed_count,
            "per_query_results": results
        }
    
    aggregated = {
        "avg_semantic_matching_score": np.mean([r["semantic_matching_score"] for r in successful_results if r]),
        "avg_hallucination_score": np.mean([r["hallucination_score"] for r in successful_results if r]),
        "avg_retrieval_score": np.mean([r["retrieval_score"] for r in successful_results if r]),
        "avg_runtime_per_query_ms": np.mean([r["runtime_per_query_ms"] for r in successful_results if r]),
        "avg_cost_per_query_usd": np.mean([r["cost_per_query_usd"] for r in successful_results if r]),
        "api_success_rate": (len(successful_results) / len(results)) * 100,
        "avg_memory_usage_mb": np.mean([r["memory_usage_mb"] for r in successful_results if r]),
        "avg_gpu_utilization_percent": np.mean([r["gpu_utilization_percent"] for r in successful_results if r]),
        "total_queries": len(qa_dataset),
        "successful_queries": len(successful_results),
        "failed_queries": failed_count,
        "per_query_results": results
    }
    
    logger.info(f"Evaluation complete. Success rate: {aggregated['api_success_rate']:.2f}%")
    return aggregated

def run_hpo_for_model(
    model_name: str,
    model_type: str,
    qa_dataset: List[Dict[str, str]],
    embeddings_data: List[Dict[str, Any]],
    index: Any,
    search_space: Dict[str, Any],
    n_trials: int = 50
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run HPO with a SINGLE model instance shared across trials"""
    
    logger.info(f"Starting Optuna HPO for {model_name} with {n_trials} trials")
    
    # Load model ONCE before trials
    logger.info(f"Loading model {model_name} (will be reused for all trials)")
    base_model = ModelFactory.create_model(
        model_name, 
        temperature=0.7,  # Dummy values, will be overridden
        top_p=0.9,
        max_tokens=500
    )
    
    # Load hallucination model once too
    hallucination_model = None
    try:
        hallucination_model = AutoModelForSequenceClassification.from_pretrained(
            "vectara/hallucination_evaluation_model", 
            trust_remote_code=True
        )
    except Exception as e:
        logger.warning(f"Could not load hallucination model: {e}")
    
    best_metrics_storage = {"metrics": None}
    
    def objective(trial: optuna.Trial) -> float:
        # Sample hyperparameters
        config = {
            "model_name": model_name,
            "model_type": model_type,
            "temperature": trial.suggest_float("temperature", 
                search_space["temperature"][0], 
                search_space["temperature"][1]),
            "top_p": trial.suggest_float("top_p", 
                search_space["top_p"][0], 
                search_space["top_p"][1]),
            "num_retrieved_docs": trial.suggest_int("num_retrieved_docs",
                search_space["num_retrieved_docs"][0],
                search_space["num_retrieved_docs"][1] - 1),
            "retrieval_method": trial.suggest_categorical("retrieval_method",
                search_space["retrieval_method"]),
            "prompt": trial.suggest_categorical("prompt",
                search_space["prompt_versions"]),
            "embedding_model": "BAAI/llm-embedder", 
        }
        
        logger.info(f"Trial {trial.number + 1}/{n_trials}")
        
        try:
            # Update sampling params instead of reloading model
            base_model.sampling_params = SamplingParams(
                temperature=config['temperature'],
                top_p=config['top_p'],
                max_tokens=base_model.max_tokens
            )

            # Evaluate with shared model instance
            metrics = evaluate_on_qa_dataset(
                qa_dataset, config, embeddings_data, 
                base_model, hallucination_model, index  # Pass pre-loaded model
            )
            
            composite_score = metrics["avg_semantic_matching_score"] * 0.5 + metrics["avg_hallucination_score"] * 0.5
            
            metrics["composite_score"] = composite_score
            metrics["config"] = config
            
            if best_metrics_storage["metrics"] is None or composite_score > best_metrics_storage["metrics"]["composite_score"]:
                best_metrics_storage["metrics"] = metrics
            
            logger.info(f"  Composite Score: {composite_score:.4f}")
            
            # Log to Optuna
            try:
                trial.set_user_attr("semantic_matching_score", metrics["avg_semantic_matching_score"])
                trial.set_user_attr("hallucination_score", metrics["avg_hallucination_score"])
            except Exception as e:
                logger.warning(f"Failed to set user attr: {e}")

            return composite_score
            
        except Exception as e:
            logger.error(f"Trial {trial.number + 1} failed: {e}")
            return -1.0
    
    # Create study and optimize
    sampler = TPESampler(seed=42, n_startup_trials=10, n_ei_candidates=24)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Cleanup after ALL trials complete
    try:
        del base_model.llm
        del base_model
        if hallucination_model is not None:
            del hallucination_model
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        import gc
        gc.collect()
        
        logger.info(f"Cleaned up {model_name}")
    except Exception as e:
        logger.warning(f"Cleanup error: {e}")
    
    # Return best config and metrics
    best_trial = study.best_trial
    best_config = {
        "model_name": model_name,
        "model_type": model_type,
        "temperature": best_trial.params["temperature"],
        "top_p": best_trial.params["top_p"],
        "num_retrieved_docs": best_trial.params["num_retrieved_docs"],
        "retrieval_method": best_trial.params["retrieval_method"],
        "prompt": best_trial.params["prompt"],
        "embedding_model": search_space.get("embedding_model", "BAAI/llm-embedder")
    }
    
    return best_config, best_metrics_storage["metrics"]

def log_model_to_mlflow(
    model_name: str,
    best_config: Dict[str, Any],
    best_metrics: Dict[str, Any],
    hpo_trials: int
):
    """
    Log best model configuration and metrics to MLflow.
    
    Args:
        model_name: Name of the model
        best_config: Best configuration from HPO
        best_metrics: Best metrics from HPO
        hpo_trials: Number of HPO trials run
    """
    with mlflow.start_run(run_name=f"{model_name}_best"):
        
        # Log parameters
        mlflow.log_param("model_name", best_config["model_name"])
        mlflow.log_param("model_type", best_config["model_type"])
        mlflow.log_param("temperature", round(best_config["temperature"], 4))
        mlflow.log_param("top_p", round(best_config["top_p"], 4))
        mlflow.log_param("num_retrieved_docs", best_config["num_retrieved_docs"])
        mlflow.log_param("retrieval_method", best_config["retrieval_method"])
        mlflow.log_param("prompt", best_config["prompt"][:200])  # Truncate if too long
        
        # Log HPO metadata
        mlflow.log_param("hpo_trials_run", hpo_trials)
        mlflow.log_param("hpo_strategy", "random_search")
        
        # Log metrics
        mlflow.log_metric("semantic_matching_score", best_metrics["avg_semantic_matching_score"])
        mlflow.log_metric("hallucination_score", best_metrics["avg_hallucination_score"])
        mlflow.log_metric("retrieval_score", best_metrics["avg_retrieval_score"])
        mlflow.log_metric("runtime_per_query_ms", best_metrics["avg_runtime_per_query_ms"])
        mlflow.log_metric("cost_per_query_usd", best_metrics["avg_cost_per_query_usd"])
        mlflow.log_metric("api_success_rate", best_metrics["api_success_rate"])
        mlflow.log_metric("memory_usage_mb", best_metrics["avg_memory_usage_mb"])
        mlflow.log_metric("gpu_utilization_percent", best_metrics["avg_gpu_utilization_percent"])
        mlflow.log_metric("composite_score", best_metrics["composite_score"])
        
        # Log tags
        mlflow.set_tags({
            "model_family": model_name.split('-')[0] if '-' in model_name else model_name,
            "is_local": str(best_config["model_type"] in ["local", "open-source"]).lower(),
            "dataset_version": "QA_v1.0",
            "experiment_date": datetime.now().strftime("%Y-%m-%d")
        })
        
        # Log artifacts
        mlflow.log_dict(best_config, "best_config.json")
        mlflow.log_text(best_config["prompt"], "prompt.txt")
        
        # Save per-query results
        per_query_file = "per_query_results.json"
        with open(per_query_file, 'w') as f:
            json.dump(best_metrics["per_query_results"], f, indent=2)
        mlflow.log_artifact(per_query_file)
        
        logger.info(f"Logged {model_name} to MLflow")

def run_mlflow_experiment(
    experiment_name: str,
    models_to_test: List[Dict[str, str]],
    qa_dataset_path: str,
    search_space: Dict[str, Any],
    n_trials_per_model: int = 20
):
    """
    Main function to run MLflow experiment for multiple models.
    
    Args:
        experiment_name: Name of MLflow experiment
        models_to_test: List of dicts with 'name' and 'type' keys
        qa_dataset_path: Path to QA JSON file
        search_space: HPO search space definition
        n_trials_per_model: Number of trials per model
    """
    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)
    
    # Load required data once
    logger.info("Loading data...")
    qa_dataset = load_qa_dataset(qa_dataset_path)
    embeddings_data = load_embeddings_data()
    index = load_faiss_index()
    
    if embeddings_data is None or index is None:
        raise Exception("Failed to load embeddings or FAISS index")
    
    # Run HPO and log for each model
    for model_info in models_to_test:
        model_name = model_info["name"]
        model_type = model_info["type"]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing model: {model_name} (type: {model_type})")
        logger.info(f"{'='*60}\n")
        
        try:
            # Run HPO
            best_config, best_metrics = run_hpo_for_model(
                model_name=model_name,
                model_type=model_type,
                qa_dataset=qa_dataset,
                embeddings_data=embeddings_data,
                index=index,
                search_space=search_space,
                n_trials=n_trials_per_model
            )
            
            # Log to MLflow
            log_model_to_mlflow(
                model_name=model_name,
                best_config=best_config,
                best_metrics=best_metrics,
                hpo_trials=n_trials_per_model
            )
            
        except Exception as e:
            logger.error(f"Failed to test model {model_name}: {e}")
            continue
    
    logger.info("\n" + "="*60)
    logger.info("Experiment complete! Check MLflow UI for results.")
    logger.info("="*60)
    clear_hf_model_cache(model_name)

# Example usage
if __name__ == "__main__":
    
    # Define models to test
    models_to_test = [
        # {"name": "gpt-3.5-turbo", "type": "openai"},
        # {"name": "gpt-4o-mini", "type": "openai"},
        # {"name": "claude-3-haiku", "type": "anthropic"},
        
        {"name": "smol_lm", "type": "open-source"},
        {"name": "llama_3.2_3b", "type": "open-source"},
        {"name": "qwen_2.5_1.5b", "type": "open-source"},
        
    ]
    # Define HPO search space
    search_space = {
        "temperature": (0.0, 1.0),
        "top_p": (0.7, 1.0),
        "num_retrieved_docs": (3, 8),  # range is [3, 8) for randint
        "retrieval_method": ["similarity", "mmr", "weighted_score", "rerank"],
        "prompt_versions": [v for _,v in PROMPTS.version.items()],
    }
    
    # Run experiment
    run_mlflow_experiment(
        experiment_name="RAG_Model_Selection",
        models_to_test=models_to_test,
        qa_dataset_path="qa.json",
        search_space=search_space,
        n_trials_per_model=20
    )