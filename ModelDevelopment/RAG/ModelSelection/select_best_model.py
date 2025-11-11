"""
select_model.py - Select best RAG model from MLflow experiments.

Uses heuristic: composite_score = semantic_score * 0.5 + hallucination_score * 0.5
"""

import json
import logging
from typing import Any, Dict, Optional

from mlflow.tracking import MlflowClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelSelector:
    """Select best model from MLflow experiment results."""
    
    def __init__(self, experiment_name: str = "RAG_Model_Selection"):
        """
        Initialize model selector.
        
        Args:
            experiment_name: Name of MLflow experiment to query
        """
        self.experiment_name = experiment_name
        self.client = MlflowClient()
        self.experiment = None
        self.runs = None
        
    def load_experiment(self) -> bool:
        """
        Load MLflow experiment.
        
        Returns:
            True if experiment loaded successfully, False otherwise
        """
        try:
            self.experiment = self.client.get_experiment_by_name(
                self.experiment_name
            )
            
            if self.experiment is None:
                logger.error(
                    f"Experiment '{self.experiment_name}' not found"
                )
                logger.info("Available experiments:")
                for exp in self.client.search_experiments():
                    logger.info(f"  - {exp.name}")
                return False
            
            logger.info(f"Loaded experiment: {self.experiment_name}")
            logger.info(f"Experiment ID: {self.experiment.experiment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading experiment: {e}")
            return False
    
    def get_all_runs(self) -> list:
        """
        Retrieve all runs from experiment ordered by composite score.
        
        Returns:
            List of MLflow run objects
        """
        try:
            self.runs = self.client.search_runs(
                experiment_ids=[self.experiment.experiment_id],
                filter_string="",
                order_by=["metrics.composite_score DESC"]
            )
            
            if not self.runs:
                logger.warning("No runs found in experiment")
                return []
            
            logger.info(f"Retrieved {len(self.runs)} runs")
            return self.runs
            
        except Exception as e:
            logger.error(f"Error retrieving runs: {e}")
            return []
    
    def calculate_composite_score(self, run) -> float:
        """
        Calculate composite score from individual metrics.
        
        Args:
            run: MLflow run object
            
        Returns:
            Composite score (semantic * 0.5 + hallucination * 0.5)
        """
        metrics = run.data.metrics
        
        # Check if composite score already exists
        if "composite_score" in metrics:
            return metrics["composite_score"]
        
        # Calculate from individual metrics
        semantic_score = metrics.get("semantic_matching_score", 0.0)
        hallucination_score = metrics.get("hallucination_score", 0.0)
        
        return semantic_score * 0.5 + hallucination_score * 0.5
    
    def select_best_model(self) -> Optional[Dict[str, Any]]:
        """
        Select best model based on composite score.
        
        Returns:
            Dictionary containing best model info and metrics, or None
        """
        if not self.runs:
            logger.error("No runs available for selection")
            return None
        
        # Get best run (already sorted by composite_score DESC)
        best_run = self.runs[0]
        
        # Extract metrics, params, and info
        metrics = best_run.data.metrics
        params = best_run.data.params
        tags = best_run.data.tags
        run_info = best_run.info
        
        # Calculate composite score if not present
        composite_score = self.calculate_composite_score(best_run)
        
        # Extract key information
        best_model_info = {
            "run_id": run_info.run_id,
            "model_name": params.get("model_name", "Unknown"),
            "model_type": params.get("model_type", "Unknown"),
            "composite_score": composite_score,
            "semantic_matching_score": metrics.get(
                "semantic_matching_score", 0.0
            ),
            "hallucination_score": metrics.get("hallucination_score", 0.0),
            "retrieval_score": metrics.get("retrieval_score", 0.0),
            "runtime_per_query_ms": metrics.get("runtime_per_query_ms", 0.0),
            "cost_per_query_usd": metrics.get("cost_per_query_usd", 0.0),
            "api_success_rate": metrics.get("api_success_rate", 0.0),
            "memory_usage_mb": metrics.get("memory_usage_mb", 0.0),
            "gpu_utilization_percent": metrics.get(
                "gpu_utilization_percent", 0.0
            ),
            "hyperparameters": {
                "temperature": params.get("temperature", None),
                "top_p": params.get("top_p", None),
                "num_retrieved_docs": params.get("num_retrieved_docs", None),
                "retrieval_method": params.get("retrieval_method", None),
            },
            "tags": {
                k: v for k, v in tags.items() 
                if not k.startswith("mlflow.")
            }
        }
        
        return best_model_info
    
    def display_results(self, best_model: Dict[str, Any]):
        """
        Display best model results in a formatted way.
        
        Args:
            best_model: Best model information dictionary
        """
        separator = "=" * 80
        
        print(f"\n{separator}")
        print("BEST MODEL SELECTED")
        print(separator)
        
        print(f"\nModel Name:           {best_model['model_name']}")
        print(f"Model Type:           {best_model['model_type']}")
        print(f"Run ID:               {best_model['run_id']}")
        
        print(f"\nCOMPOSITE SCORE:      "
              f"{best_model['composite_score']:.4f}")
        print(f"  Semantic Score:     "
              f"{best_model['semantic_matching_score']:.4f} (50%)")
        print(f"  Hallucination:      "
              f"{best_model['hallucination_score']:.4f} (50%)")
        
        print("\nOTHER METRICS:")
        print(f"  Retrieval Score:    "
              f"{best_model['retrieval_score']:.4f}")
        print(f"  Runtime per Query:  "
              f"{best_model['runtime_per_query_ms']:.2f} ms")
        print(f"  Cost per Query:     "
              f"${best_model['cost_per_query_usd']:.6f}")
        print(f"  API Success Rate:   "
              f"{best_model['api_success_rate']:.2f}%")
        print(f"  Memory Usage:       "
              f"{best_model['memory_usage_mb']:.2f} MB")
        print(f"  GPU Utilization:    "
              f"{best_model['gpu_utilization_percent']:.2f}%")
        
        print("\nHYPERPARAMETERS:")
        for param, value in best_model['hyperparameters'].items():
            if value is not None:
                print(f"  {param:20} {value}")
        
        if best_model['tags']:
            print("\nTAGS:")
            for tag_key, tag_value in best_model['tags'].items():
                print(f"  {tag_key}: {tag_value}")
        
        print(f"{separator}\n")
    
    def save_best_model_config(
        self, 
        best_model: Dict[str, Any]
    ):
        """
        Save best model configuration to JSON file.
        
        Args:
            best_model: Best model information dictionary
            output_path: Path to save configuration file
        """
        try:
            # Get the full run to access params
            run = self.client.get_run(best_model["run_id"])
            params = run.data.params
            
            # Get prompt and embedding model from params
            prompt_text = params.get("prompt", "")
            embedding_model = params.get(
                "embedding_model", 
                "BAAI/llm-embedder"
            )
            
            # Create config for deployment
            # (compatible with run_rag_pipeline)
            deployment_config = {
                # Model configuration
                "model_name": best_model["model_name"],
                "model_type": best_model["model_type"],
                "temperature": float(params.get("temperature", 0.7)),
                "top_p": float(params.get("top_p", 0.9)),
                
                # RAG pipeline configuration
                "k": int(params.get("num_retrieved_docs", 5)),
                "retrieval_method": params.get(
                    "retrieval_method", 
                    "similarity"
                ),
                "embedding_model": embedding_model,
                "prompt": prompt_text,
                
                # MLflow tracking
                "mlflow_run_id": best_model["run_id"],
                
                # Performance metrics
                "performance_metrics": {
                    "composite_score": best_model["composite_score"],
                    "semantic_matching_score": best_model[
                        "semantic_matching_score"
                    ],
                    "hallucination_score": best_model["hallucination_score"],
                    "retrieval_score": best_model["retrieval_score"],
                    "api_success_rate": best_model["api_success_rate"]
                },
                
                # Resource requirements
                "resource_requirements": {
                    "memory_mb": best_model["memory_usage_mb"],
                    "gpu_utilization_percent": best_model[
                        "gpu_utilization_percent"
                    ]
                },
                
                # Cost metrics
                "cost_metrics": {
                    "cost_per_query_usd": best_model["cost_per_query_usd"],
                    "runtime_per_query_ms": best_model[
                        "runtime_per_query_ms"
                    ]
                },
                
                # Tags
                "tags": best_model["tags"]
            }

            import os
            import sys
            cur_dir = os.path.dirname(__file__) #ModelSelection
            parent_dir = os.path.dirname(cur_dir) #RAG
            sys.path.insert(0, parent_dir)

            save_path = os.path.join(parent_dir, "utils")
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)

            save_path = os.path.join(save_path, "RAG_config.json")

            with open(save_path, 'w') as f:
                json.dump(deployment_config, f, indent=2)
            
            logger.info(f"Best model config saved to: {save_path}")
            logger.info(
                "Config contains all fields needed by run_rag_pipeline()"
            )
            
        except Exception as e:
            logger.error(f"Error saving config: {e}")


def main():
    """Main function to select best model."""
    
    # Initialize selector
    selector = ModelSelector(experiment_name="RAG_Model_Selection")
    
    # Load experiment
    if not selector.load_experiment():
        logger.error("Failed to load experiment. Exiting.")
        return
    
    # Get all runs
    runs = selector.get_all_runs()
    if not runs:
        logger.error("No runs found. Exiting.")
        return
    
    # Select best model
    best_model = selector.select_best_model()
    if best_model is None:
        logger.error("Could not select best model. Exiting.")
        return
    
    # Display results
    selector.display_results(best_model)
    
    # Save best model config
    selector.save_best_model_config(best_model)
    
    print("Model selection complete! "
          "Check 'utils/RAG_config.json' for deployment config.\n")


if __name__ == "__main__":
    main()