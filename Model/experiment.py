# experiment.py
import sys
import json
import random
import logging
from functools import partial
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

import optuna
import mlflow
import numpy as np
import pandas as pd

# RAG_inference imports (assumes RAG_inference.py is in same folder or importable)
from RAG.RAG_inference import (
    run_inference,
    precompute_document_embeddings,
    get_embedding_model,  # ensures model is loaded if needed
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Experiment constants
TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
TYPE = None # Either API or OS (open-source)
NUM_TRIALS = 50
METRIC_OPTIMIZATION = "semantic_matching_score"

# Enable MLflow autologging (will log parameters/metrics automatically where supported)
try:
    mlflow.autolog()
    # Optional: set tracking URI if MLflow server is used
    # mlflow.set_tracking_uri("http://mlflow-server:5000")
except Exception as e:
    logger.warning("mlflow.autolog() failed or mlflow not fully configured: %s", str(e))


# --- Data loading -------------------------------------------------------------
def load_question_bank(file_path: str) -> List[Dict[str, Any]]:
    try:
        with open(file_path, 'r') as f:
            questions = json.load(f)
        logger.info("Loaded %d questions from %s", len(questions), file_path)
        return questions
    except Exception as e:
        logger.error("Failed to load question bank: %s", str(e))
        return []

def load_document_store(file_path: str) -> Tuple[List[str], Dict[str, str]]:
    try:
        with open(file_path, 'r') as f:
            doc_store = json.load(f)
        # doc_store expected to be id->text mapping
        doc_texts = list(doc_store.values())
        doc_mapping = doc_store
        logger.info("Loaded %d documents from %s", len(doc_texts), file_path)
        return doc_texts, doc_mapping
    except Exception as e:
        logger.error("Failed to load document store: %s", str(e))
        return [], {}

# --- Prompt templates ---------------------------------------------------------
def get_prompt_template(prompt_type: str) -> str:
    """
    Placeholder for prompts. 
    TODO: WRITE BETTER PROMPTS
    """
    templates = {
        'typ_1': """Prompt""",
        'type_2': """Prompt"""
    }
    return templates.get(prompt_type, templates["basic"])

# --- Evaluation ---------------------------------------------------------------
def evaluate_model(
    questions: List[Dict[str, Any]],
    document_texts: List[str],
    doc_mapping: Dict[str, str],
    model_name: str,
    model_type: str,
    temperature: float,
    top_p: float,
    num_retrieved_docs: int,
    retrieval_method: str,
    prompt_type: str,
    model_config: Dict[str, Any],
    precomputed_doc_embeddings: Optional[Dict[int, Any]] = None,
    sample_size: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate model on `sample_size` questions (or all if None).
    Returns average metrics.
    """
    # sample questions safely (random.sample on list of dicts)
    if sample_size and sample_size < len(questions):
        sampled_questions = random.sample(questions, sample_size)
    else:
        sampled_questions = list(questions)

    logger.info("Evaluating %s (%s) on %d questions", model_name, model_type, len(sampled_questions))

    prompt_template = get_prompt_template(prompt_type)

    # accumulator
    metrics_sum = {
        'hallucination_score': 0.0,
        'retrieval_score': 0.0,
        'semantic_matching_score': 0.0,
        'avg_input_tokens': 0.0,
        'avg_output_tokens': 0.0,
        'runtime_per_query_ms': 0.0,
        'cost_per_query_usd': 0.0,
        'memory_usage_mb': 0.0,
        'api_success_rate': 0.0,
        'gpu_utilization_percent': 0.0
    }

    for i, question in enumerate(sampled_questions):
        relevant_doc_texts = [doc_mapping.get(doc_id, "") for doc_id in question.get("relevant_docs", [])]
        relevant_doc_texts = [d for d in relevant_doc_texts if d]

        # run inference
        result = run_inference(
            query=question["query"],
            document_store=document_texts,
            reference_answer=question.get("reference_answer"),
            relevant_docs=relevant_doc_texts,
            model_name=model_name,
            model_type=model_type,
            temperature=temperature,
            top_p=top_p,
            num_retrieved_docs=num_retrieved_docs,
            retrieval_method=retrieval_method,
            prompt_type=prompt_type,
            prompt_template=prompt_template,
            model_config=model_config,
            doc_embeddings=precomputed_doc_embeddings
        )

        # accumulate
        for k, v in result["metrics"].items():
            if k in metrics_sum:
                metrics_sum[k] += v

        if (i + 1) % 5 == 0:
            logger.info("  Processed %d/%d questions", i + 1, len(sampled_questions))

    num = len(sampled_questions)
    avg_metrics = {k: (v / num) for k, v in metrics_sum.items()}
    logger.info("Avg semantic score: %.4f", avg_metrics["semantic_matching_score"])
    return avg_metrics

# --- Optuna objective --------------------------------------------------------
def objective(
    trial: optuna.Trial,
    questions: List[Dict[str, Any]],
    document_texts: List[str],
    doc_mapping: Dict[str, str],
    model_name: str,
    model_type: str,
    fixed_params: Dict[str, Any] = None,
    precomputed_doc_embeddings: Optional[Dict[int, Any]] = None
) -> float:
    mlflow.start_run(nested=True)
    try:
        # sample hyperparameters
        temperature = trial.suggest_categorical("temperature", [0.1, 0.3, 0.5, 0.7, 1.0])
        top_p = trial.suggest_categorical("top_p", [0.9, 0.95, 1.0])
        num_retrieved_docs = trial.suggest_categorical("num_retrieved_docs", [1, 3, 5, 10])
        retrieval_method = trial.suggest_categorical("retrieval_method", ["bm25", "embedding", "hybrid"])
        prompt_type = trial.suggest_categorical("prompt_type", ["basic", "cot", "with_examples", "refine"])

        # get model config from fixed_params if supplied
        model_config = fixed_params.get("model_config", {}) if fixed_params else {}

        # log params to mlflow
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("temperature", temperature)
        mlflow.log_param("top_p", top_p)
        mlflow.log_param("num_retrieved_docs", num_retrieved_docs)
        mlflow.log_param("retrieval_method", retrieval_method)
        mlflow.log_param("prompt_type", prompt_type)

        # evaluate on small sample for speed
        sample_size = min(10, len(questions))
        avg_metrics = evaluate_model(
            questions=questions,
            document_texts=document_texts,
            doc_mapping=doc_mapping,
            model_name=model_name,
            model_type=model_type,
            temperature=temperature,
            top_p=top_p,
            num_retrieved_docs=num_retrieved_docs,
            retrieval_method=retrieval_method,
            prompt_type=prompt_type,
            model_config=model_config,
            precomputed_doc_embeddings=precomputed_doc_embeddings,
            sample_size=sample_size
        )

        for metric, val in avg_metrics.items():
            mlflow.log_metric(metric, val)

        # objective (maximize semantic score by default)
        target_metric = avg_metrics.get(METRIC_OPTIMIZATION, 0.0)

        # If optimizing hallucination (lower better) we invert
        if METRIC_OPTIMIZATION == "hallucination_score":
            return -target_metric

        return target_metric
    finally:
        mlflow.end_run()

# --- Run experiment ----------------------------------------------------------
# Modified run_experiment function to focus on a single model
def run_experiment(
    question_bank_path: str,
    document_store_path: str,
    model_config: Dict[str, Any] = {},
    experiment_name: Optional[str] = None,
    n_trials: int = NUM_TRIALS,
    sample_size: Optional[int] = None,
    precompute_embeddings: bool = True
):
    try:
        model_name = model_config.get('model_name', None)
        model_type = model_config.get('model_type', None)
        if model_name == None or model_type == None:
            raise ("Model name: {}\nModel type: {}\n\nOne of them is None. Both must be a valid string")
    except Exception as e:
        logger.error(e)
        sys.exit(1)
    if experiment_name is None:
        experiment_name = f"RAG_Experiment_{model_type}_{model_name}"
    mlflow.set_experiment(experiment_name)
    logger.info("Starting experiment: %s", experiment_name)

    questions = load_question_bank(question_bank_path)
    document_texts, doc_mapping = load_document_store(document_store_path)
    if not questions or not document_texts:
        logger.error("Missing data. Exiting.")
        return

    # Optionally precompute document embeddings (recommended)
    precomputed_doc_embeddings = None
    if precompute_embeddings:
        precomputed_doc_embeddings = precompute_document_embeddings(document_texts, force=False)

    logger.info("Optimizing for model %s (%s)", model_name, model_type)

    study_name = f"{model_type}_{model_name}_{TIMESTAMP}"
    study = optuna.create_study(direction="maximize", study_name=study_name)

    objective_func = partial(
        objective,
        questions=questions,
        document_texts=document_texts,
        doc_mapping=doc_mapping,
        model_name=model_name,
        model_type=model_type,
        fixed_params={"model_config": model_config},
        precomputed_doc_embeddings=precomputed_doc_embeddings
    )

    # Create a parent run for the entire HPO process
    with mlflow.start_run(run_name=f"{model_type}_{model_name}_hpo_{TIMESTAMP}"):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_type", model_type)
        for k, v in model_config.items():
            mlflow.log_param(f"config_{k}", v)
        
        # Log HPO parameters
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("optimization_metric", METRIC_OPTIMIZATION)
        
        # Run optimization
        study.optimize(objective_func, n_trials=n_trials)
        
        # Log best trial info in parent run
        mlflow.log_param("best_trial_number", study.best_trial.number)
        mlflow.log_metric("best_trial_value", study.best_value)
        for param_name, param_value in study.best_params.items():
            mlflow.log_param(f"best_{param_name}", param_value)
        
        # Full evaluation with best params
        best_params = study.best_params
        prompt_type = best_params.get("prompt_type", "basic")

        logger.info("HPO complete. Best trial: %d, value: %.4f", 
                   study.best_trial.number, study.best_value)
        logger.info("Best params: %s", study.best_params)

        # Create a child run for the final evaluation with best params
        with mlflow.start_run(run_name=f"{model_type}_{model_name}_best_{TIMESTAMP}", nested=True):
            for k, v in best_params.items():
                mlflow.log_param(k, v)

            eval_sample_size = sample_size if sample_size else len(questions)
            avg_metrics = evaluate_model(
                questions=questions,
                document_texts=document_texts,
                doc_mapping=doc_mapping,
                model_name=model_name,
                model_type=model_type,
                temperature=best_params.get("temperature", 0.7),
                top_p=best_params.get("top_p", 0.95),
                num_retrieved_docs=best_params.get("num_retrieved_docs", 5),
                retrieval_method=best_params.get("retrieval_method", "hybrid"),
                prompt_type=prompt_type,
                model_config=model_config,
                precomputed_doc_embeddings=precomputed_doc_embeddings,
                sample_size=eval_sample_size
            )

            # Log evaluation metrics for best configuration
            for metric, value in avg_metrics.items():
                mlflow.log_metric(metric, value)
            
            # Create and log a model artifact (optional)
            model_info = {
                "model_name": model_name,
                "model_type": model_type,
                "parameters": best_params,
                "metrics": avg_metrics
            }
            
            # Log the best model configuration as a JSON artifact
            with open("best_model_config.json", "w") as f:
                json.dump(model_info, f, indent=2)
            mlflow.log_artifact("best_model_config.json")
            
        # Print summary
        print("\n=== Experiment Summary ===")
        print(f"Model: {model_name} ({model_type})")
        print(f"Best params: {best_params}")
        for metric, val in avg_metrics.items():
            print(f"  {metric}: {val:.4f}")

    logger.info("Experiment complete.")

# --- Example usage -----------------------------------------------------------
if __name__ == "__main__":
    model_configs = None
    # list of API models or OS models
    # example - 
    #[{
    #         "model_name": "gpt-4",
    #         "model_type": "api",
    #         "model_config": {}
    #     }, 
    #     {
    #         "model_name": "llama-3.1-8b",
    #         "model_type": "local",
    #         "model_config": {"device": "cuda"}
    #     }]

    # Update these paths to your actual data files
    question_bank_path = "data/question_bank.json"
    document_store_path = "data/document_store.json"

    # Run experiment for a single model
    for config in model_configs:
        run_experiment( 
            question_bank_path=question_bank_path,
            document_store_path=document_store_path,
            model_config=config,
            experiment_name=None,
            n_trials=20,           # reduce during testing; increase for full runs
            sample_size=20,        # final evaluation sample size (or None for all)
            precompute_embeddings=True
        )