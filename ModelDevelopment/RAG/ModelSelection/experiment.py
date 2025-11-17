"""
experiment.py - RAG model selection with HPO
Reads data from GCS, runs experiments, logs to MLflow
"""
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
from vllm import SamplingParams
from rag_bias_adapter import run_bias_check

cur_dir = os.path.dirname(__file__)
path = os.path.dirname(cur_dir)
sys.path.insert(0, path)

from models.models import ModelFactory
from ModelInference.RAG_inference import (
    load_embeddings_data,
    load_faiss_index,
    get_embedding,
    compute_hallucination_score
)
from transformers import AutoModelForSequenceClassification
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

QA_PATH = os.path.join(cur_dir, "qa.json")


def clear_gpu_memory():
    """Clear GPU memory and cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()


def generate_response(
    query: str,
    documents: List[Dict[str, Any]],
    config: Dict[str, Any],
    model: Any
) -> Optional[Tuple[str, int, int]]:
    """Generate response using LLM"""
    try:
        model_name = config.get("model_name")
        prompt_template = config.get("prompt")
        
        if not model_name or not prompt_template:
            raise ValueError("Missing model_name or prompt in config")
        
        context_parts = []
        for d in documents:
            title = d.get('title', 'Unknown')
            content = d.get('content', '')
            context_parts.append(f"Document {d['rank']} - {title}:\n{content}")
        
        context = "\n\n".join(context_parts)
        formatted_prompt = prompt_template.format(context=context, query=query)
        
        response_d = model.infer(formatted_prompt)
        
        if not response_d or not response_d.get('success'):
            raise ValueError(f"Model inference failed: {response_d}")
        
        response = response_d.get('generated_text', '')
        in_tokens = response_d.get('input_tokens', 0)
        out_tokens = response_d.get('output_tokens', 0)
        
        references = "\n\n**References:**\n"
        for d in documents:
            link = d.get('metadata', {}).get('link', '')
            title = d.get('title', 'Unknown')
            if link:
                references += f"{d['rank']}. [{title}]({link})\n"
            else:
                references += f"{d['rank']}. {title}\n"
        
        response += references
        
        return response, in_tokens, out_tokens
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return None


def compute_answer_relevance(generated_answer: str, reference_answer: str) -> Dict[str, float]:
    """Compute keyword-based relevance metrics"""
    
    def preprocess(text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'of', 'with', 'by', 'from', 'is', 'was',
            'are', 'were', 'been', 'be', 'have', 'has', 'had'
        }
        
        return [w for w in words if w not in stop_words and len(w) > 2]
    
    def get_bigrams(words: List[str]) -> List[Tuple[str, str]]:
        return [(words[i], words[i+1]) for i in range(len(words)-1)]
    
    gen_words = preprocess(generated_answer)
    ref_words = preprocess(reference_answer)
    
    if not gen_words or not ref_words:
        return {
            "keyword_precision": 0.0,
            "keyword_recall": 0.0,
            "keyword_f1": 0.0,
            "unigram_overlap": 0.0,
            "bigram_overlap": 0.0
        }
    
    gen_set = set(gen_words)
    ref_set = set(ref_words)
    
    common_words = gen_set.intersection(ref_set)
    
    precision = len(common_words) / len(gen_set) if gen_set else 0.0
    recall = len(common_words) / len(ref_set) if ref_set else 0.0
    
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    
    union = gen_set.union(ref_set)
    unigram_jaccard = len(common_words) / len(union) if union else 0.0
    
    gen_bigrams = set(get_bigrams(gen_words))
    ref_bigrams = set(get_bigrams(ref_words))
    
    if gen_bigrams and ref_bigrams:
        common_bigrams = gen_bigrams.intersection(ref_bigrams)
        bigram_union = gen_bigrams.union(ref_bigrams)
        bigram_jaccard = len(common_bigrams) / len(bigram_union) if bigram_union else 0.0
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
    """Load QA dataset"""
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
    """Evaluate RAG pipeline on single query"""
    try:
        query = qa_pair["Q"]
        reference_answer = qa_pair["A"]
        
        start_time = time.time()
        
        embedding = get_embedding(query, "BAAI/llm-embedder")
        if embedding is None:
            raise Exception("Failed to generate embedding")
        
        k = config.get("num_retrieved_docs", 5)
        retrieval_method = config.get("retrieval_method", "similarity")
        documents = retrieve_documents(embedding, index, embeddings_data, k, retrieval_method)
        if not documents:
            raise Exception("Failed to retrieve documents")
        
        result = generate_response(query, documents, config, model)
        if result is None:
            raise Exception("Failed to generate response")
        
        response, in_tokens, out_tokens = result
        
        runtime_ms = (time.time() - start_time) * 1000
        
        relevance_metrics = compute_answer_relevance(response, reference_answer)
        semantic_matching_score = relevance_metrics["keyword_f1"]
        
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
        
        retrieval_scores = [doc.get("score", 0.0) for doc in documents]
        avg_retrieval_score = sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0.0
        
        cost_per_query = 0.0
        
        return {
            "query": query,
            "reference_answer": reference_answer,
            "generated_answer": response,
            "semantic_matching_score": semantic_matching_score,
            "hallucination_score": avg_hallucination_score,
            "retrieval_score": avg_retrieval_score,
            "runtime_per_query_ms": runtime_ms,
            "cost_per_query_usd": cost_per_query,
            "api_success_rate": 100.0,
            "memory_usage_mb": 0.0,
            "gpu_utilization_percent": 0.0,
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "relevance_details": relevance_metrics,
            "num_retrieved_docs": len(documents)
        }
        
    except Exception as e:
        logger.error(f"Error evaluating query: {e}")
        return {
            "query": qa_pair.get("Q", ""),
            "error": str(e),
            "semantic_matching_score": 0.0,
            "hallucination_score": 0.0,
            "retrieval_score": 0.0,
            "runtime_per_query_ms": 0.0,
            "cost_per_query_usd": 0.0,
            "api_success_rate": 0.0
        }


def evaluate_on_qa_dataset(
    qa_dataset: List[Dict[str, str]],
    config: Dict[str, Any],
    embeddings_data: List[Dict[str, Any]],
    model: Any,
    hallucination_model: Any,
    index: Any
) -> Dict[str, Any]:
    """Evaluate on entire QA dataset"""
    logger.info(f"Evaluating on {len(qa_dataset)} QA pairs")
    
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
    
    try:
        del model
        del hallucination_model
        clear_gpu_memory()
        import gc
        gc.collect()
    except Exception as e:
        logger.warning(f"Cleanup error: {e}")
    
    successful_results = [r for r in results if "error" not in r.keys()]
    failed_count = len(results) - len(successful_results)
    
    if not successful_results:
        logger.error("All queries failed")
        return {
            "avg_semantic_matching_score": 0.0,
            "avg_hallucination_score": 0.0,
            "avg_retrieval_score": 0.0,
            "avg_runtime_per_query_ms": 0.0,
            "avg_cost_per_query_usd": 0.0,
            "api_success_rate": 0.0,
            "total_queries": len(qa_dataset),
            "failed_queries": failed_count,
            "per_query_results": results
        }
    
    aggregated = {
        "avg_semantic_matching_score": np.mean([r["semantic_matching_score"] for r in successful_results]),
        "avg_hallucination_score": np.mean([r["hallucination_score"] for r in successful_results]),
        "avg_retrieval_score": np.mean([r["retrieval_score"] for r in successful_results]),
        "avg_runtime_per_query_ms": np.mean([r["runtime_per_query_ms"] for r in successful_results]),
        "avg_cost_per_query_usd": np.mean([r["cost_per_query_usd"] for r in successful_results]),
        "api_success_rate": (len(successful_results) / len(results)) * 100,
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
    n_trials: int = 20
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run HPO with single model instance"""
    
    logger.info(f"Starting Optuna HPO for {model_name} ({n_trials} trials)")
    
    logger.info(f"Loading model: {model_name}")
    base_model = ModelFactory.create_model(
        model_name,
        temperature=0.7,
        top_p=0.9,
        max_tokens=500
    )
    
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
            base_model.sampling_params = SamplingParams(
                temperature=config['temperature'],
                top_p=config['top_p'],
                max_tokens=base_model.max_tokens
            )
            
            metrics = evaluate_on_qa_dataset(
                qa_dataset, config, embeddings_data,
                base_model, hallucination_model, index
            )
            
            composite_score = metrics["avg_semantic_matching_score"] * 0.5 + metrics["avg_hallucination_score"] * 0.5
            
            metrics["composite_score"] = composite_score
            metrics["config"] = config
            
            if best_metrics_storage["metrics"] is None or composite_score > best_metrics_storage["metrics"]["composite_score"]:
                best_metrics_storage["metrics"] = metrics
            
            logger.info(f"Composite Score: {composite_score:.4f}")
            
            try:
                trial.set_user_attr("semantic_matching_score", metrics["avg_semantic_matching_score"])
                trial.set_user_attr("hallucination_score", metrics["avg_hallucination_score"])
            except Exception as e:
                logger.warning(f"Failed to set user attr: {e}")
            
            return composite_score
            
        except Exception as e:
            logger.error(f"Trial {trial.number + 1} failed: {e}")
            return -1.0
    
    sampler = TPESampler(seed=42, n_startup_trials=10, n_ei_candidates=24)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    try:
        del base_model.llm
        del base_model
        if hallucination_model is not None:
            del hallucination_model
        
        clear_gpu_memory()
        import gc
        gc.collect()
    except Exception as e:
        logger.warning(f"Cleanup error: {e}")
    
    best_trial = study.best_trial
    best_config = {
        "model_name": model_name,
        "model_type": model_type,
        "temperature": best_trial.params["temperature"],
        "top_p": best_trial.params["top_p"],
        "num_retrieved_docs": best_trial.params["num_retrieved_docs"],
        "retrieval_method": best_trial.params["retrieval_method"],
        "prompt": best_trial.params["prompt"],
        "embedding_model": "BAAI/llm-embedder"
    }
    
    return best_config, best_metrics_storage["metrics"]


def log_model_to_mlflow(
    model_name: str,
    best_config: Dict[str, Any],
    best_metrics: Dict[str, Any],
    hpo_trials: int
):
    """Log best model to MLflow"""
    with mlflow.start_run(run_name=f"{model_name}_best"):
        
        mlflow.log_param("model_name", best_config["model_name"])
        mlflow.log_param("model_type", best_config["model_type"])
        mlflow.log_param("temperature", round(best_config["temperature"], 4))
        mlflow.log_param("top_p", round(best_config["top_p"], 4))
        mlflow.log_param("num_retrieved_docs", best_config["num_retrieved_docs"])
        mlflow.log_param("retrieval_method", best_config["retrieval_method"])
        mlflow.log_param("prompt", best_config["prompt"][:200])
        mlflow.log_param("hpo_trials_run", hpo_trials)
        mlflow.log_param("hpo_strategy", "optuna_tpe")
        
        mlflow.log_metric("semantic_matching_score", best_metrics["avg_semantic_matching_score"])
        mlflow.log_metric("hallucination_score", best_metrics["avg_hallucination_score"])
        mlflow.log_metric("retrieval_score", best_metrics["avg_retrieval_score"])
        mlflow.log_metric("runtime_per_query_ms", best_metrics["avg_runtime_per_query_ms"])
        mlflow.log_metric("cost_per_query_usd", best_metrics["avg_cost_per_query_usd"])
        mlflow.log_metric("api_success_rate", best_metrics["api_success_rate"])
        mlflow.log_metric("composite_score", best_metrics["composite_score"])
        
        mlflow.set_tags({
            "model_family": model_name.split('_')[0],
            "is_local": str(best_config["model_type"] in ["local", "open-source"]).lower(),
            "dataset_version": "QA_v1.0",
            "experiment_date": datetime.now().strftime("%Y-%m-%d")
        })
        
        mlflow.log_dict(best_config, "best_config.json")
        mlflow.log_text(best_config["prompt"], "prompt.txt")
        
        per_query_file = "per_query_results.json"
        with open(per_query_file, 'w') as f:
            json.dump(best_metrics["per_query_results"], f, indent=2)
        mlflow.log_artifact(per_query_file)
        
        logger.info(f"Logged {model_name} to MLflow")
        
        try:
            logger.info("Running bias detection")
            run_bias_check(best_metrics)
        except Exception as e:
            logger.warning(f"Bias detection failed: {e}")


def run_mlflow_experiment(
    experiment_name: str,
    models_to_test: List[Dict[str, str]],
    qa_dataset_path: str,
    search_space: Dict[str, Any],
    n_trials_per_model: int = 20
):
    """Run MLflow experiment for model selection"""
    mlflow.set_experiment(experiment_name)
    
    logger.info("Loading data from GCS/local")
    qa_dataset = load_qa_dataset(qa_dataset_path)
    embeddings_data = load_embeddings_data()
    index = load_faiss_index()
    
    if embeddings_data is None or index is None:
        raise Exception("Failed to load embeddings or FAISS index")
    
    for model_info in models_to_test:
        model_name = model_info["name"]
        model_type = model_info["type"]
        
        logger.info("=" * 60)
        logger.info(f"Testing: {model_name} ({model_type})")
        logger.info("=" * 60)
        
        try:
            best_config, best_metrics = run_hpo_for_model(
                model_name=model_name,
                model_type=model_type,
                qa_dataset=qa_dataset,
                embeddings_data=embeddings_data,
                index=index,
                search_space=search_space,
                n_trials=n_trials_per_model
            )
            
            log_model_to_mlflow(
                model_name=model_name,
                best_config=best_config,
                best_metrics=best_metrics,
                hpo_trials=n_trials_per_model
            )
            
        except Exception as e:
            logger.error(f"Failed to test {model_name}: {e}")
            continue
    
    logger.info("=" * 60)
    logger.info("Experiment complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    
    models_to_test = [
        {"name": "smol_lm", "type": "open-source"},
        {"name": "qwen_2.5_1.5b", "type": "open-source"},
        {"name": "llama_3.2_3b", "type": "open-source"},
    ]
    
    search_space = {
        "temperature": (0.0, 1.0),
        "top_p": (0.7, 1.0),
        "num_retrieved_docs": (3, 8),
        "retrieval_method": ["similarity", "mmr", "weighted_score", "rerank"],
        "prompt_versions": [v for _, v in PROMPTS.version.items()],
    }
    
    run_mlflow_experiment(
        experiment_name="RAG_Model_Selection",
        models_to_test=models_to_test,
        qa_dataset_path=QA_PATH,
        search_space=search_space,
        n_trials_per_model=20
    )