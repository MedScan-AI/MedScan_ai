"""
select_model.py - Select best RAG model from MLflow experiments
Uses heuristic: composite_score = semantic_score * 0.5 + hallucination_score * 0.5
"""

import logging
import mlflow
import pandas as pd
from typing import Dict, Any, List, Optional
import json
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelSelector:
    """Select best model from MLflow experiment results"""
    
    def __init__(self, experiment_name: str = "RAG_Model_Selection"):
        """
        Initialize model selector
        
        Args:
            experiment_name: Name of MLflow experiment to query
        """
        self.experiment_name = experiment_name
        self.experiment = None
        self.runs_df = None
        
    def load_experiment(self) -> bool:
        """Load MLflow experiment"""
        try:
            self.experiment = mlflow.get_experiment_by_name(self.experiment_name)
            
            if self.experiment is None:
                logger.error(f"Experiment '{self.experiment_name}' not found")
                logger.info("Available experiments:")
                for exp in mlflow.search_experiments():
                    logger.info(f"  - {exp.name}")
                return False
            
            logger.info(f"Loaded experiment: {self.experiment_name}")
            logger.info(f"Experiment ID: {self.experiment.experiment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading experiment: {e}")
            return False
    
    def get_all_runs(self) -> pd.DataFrame:
        """Retrieve all runs from experiment"""
        try:
            runs = mlflow.search_runs(
                experiment_ids=[self.experiment.experiment_id],
                order_by=["metrics.composite_score DESC"]
            )
            
            if runs.empty:
                logger.warning("No runs found in experiment")
                return pd.DataFrame()
            
            logger.info(f"Retrieved {len(runs)} runs")
            self.runs_df = runs
            return runs
            
        except Exception as e:
            logger.error(f"Error retrieving runs: {e}")
            return pd.DataFrame()
    
    def calculate_composite_scores(self, runs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate composite scores if not already present
        
        Args:
            runs_df: DataFrame of MLflow runs
            
        Returns:
            DataFrame with composite_score column added
        """
        if 'metrics.composite_score' not in runs_df.columns:
            logger.info("Calculating composite scores from individual metrics")
            
            semantic_col = 'metrics.semantic_matching_score'
            hallucination_col = 'metrics.hallucination_score'
            
            if semantic_col in runs_df.columns and hallucination_col in runs_df.columns:
                runs_df['metrics.composite_score'] = (
                    runs_df[semantic_col] * 0.5 + 
                    runs_df[hallucination_col] * 0.5
                )
            else:
                logger.error("Required metrics not found in runs")
                return runs_df
        
        return runs_df
    
    def select_best_model(self) -> Optional[Dict[str, Any]]:
        """
        Select best model based on composite score
        
        Returns:
            Dictionary containing best model info and metrics
        """
        if self.runs_df is None or self.runs_df.empty:
            logger.error("No runs available for selection")
            return None
        
        # Calculate composite scores if needed
        self.runs_df = self.calculate_composite_scores(self.runs_df)
        
        # Sort by composite score (descending)
        sorted_runs = self.runs_df.sort_values(
            'metrics.composite_score', 
            ascending=False
        )
        
        if sorted_runs.empty:
            logger.error("No valid runs with composite scores")
            return None
        
        # Get best run
        best_run = sorted_runs.iloc[0]
        
        # Extract key information
        best_model_info = {
            "run_id": best_run["run_id"],
            "model_name": best_run.get("params.model_name", "Unknown"),
            "model_type": best_run.get("params.model_type", "Unknown"),
            "composite_score": best_run["metrics.composite_score"],
            "semantic_matching_score": best_run.get("metrics.semantic_matching_score", 0.0),
            "hallucination_score": best_run.get("metrics.hallucination_score", 0.0),
            "retrieval_score": best_run.get("metrics.retrieval_score", 0.0),
            "runtime_per_query_ms": best_run.get("metrics.runtime_per_query_ms", 0.0),
            "cost_per_query_usd": best_run.get("metrics.cost_per_query_usd", 0.0),
            "api_success_rate": best_run.get("metrics.api_success_rate", 0.0),
            "memory_usage_mb": best_run.get("metrics.memory_usage_mb", 0.0),
            "gpu_utilization_percent": best_run.get("metrics.gpu_utilization_percent", 0.0),
            "hyperparameters": {
                "temperature": best_run.get("params.temperature", None),
                "top_p": best_run.get("params.top_p", None),
                "num_retrieved_docs": best_run.get("params.num_retrieved_docs", None),
                "retrieval_method": best_run.get("params.retrieval_method", None),
            }
        }
        
        return best_model_info
    
    def get_top_n_models(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Get top N models by composite score
        
        Args:
            n: Number of top models to return
            
        Returns:
            List of model info dictionaries
        """
        if self.runs_df is None or self.runs_df.empty:
            return []
        
        # Calculate composite scores if needed
        self.runs_df = self.calculate_composite_scores(self.runs_df)
        
        # Sort and get top N
        sorted_runs = self.runs_df.sort_values(
            'metrics.composite_score', 
            ascending=False
        ).head(n)
        
        top_models = []
        for _, run in sorted_runs.iterrows():
            model_info = {
                "rank": len(top_models) + 1,
                "model_name": run.get("params.model_name", "Unknown"),
                "composite_score": run["metrics.composite_score"],
                "semantic_score": run.get("metrics.semantic_matching_score", 0.0),
                "hallucination_score": run.get("metrics.hallucination_score", 0.0),
                "runtime_ms": run.get("metrics.runtime_per_query_ms", 0.0),
                "cost_usd": run.get("metrics.cost_per_query_usd", 0.0),
            }
            top_models.append(model_info)
        
        return top_models
    
    def display_results(self, best_model: Dict[str, Any]):
        """
        Display best model results in a formatted way
        
        Args:
            best_model: Best model information
        """
        print("\n" + "="*80)
        print("🏆 BEST MODEL SELECTED")
        print("="*80)
        
        print(f"\nModel Name:           {best_model['model_name']}")
        print(f"Model Type:           {best_model['model_type']}")
        print(f"Run ID:               {best_model['run_id']}")
        print(f"\n📊 COMPOSITE SCORE:    {best_model['composite_score']:.4f}")
        print(f"   └─ Semantic Score:  {best_model['semantic_matching_score']:.4f} (50%)")
        print(f"   └─ Hallucination:   {best_model['hallucination_score']:.4f} (50%)")
        
        print(f"\n📈 OTHER METRICS:")
        print(f"   Retrieval Score:    {best_model['retrieval_score']:.4f}")
        print(f"   Runtime per Query:  {best_model['runtime_per_query_ms']:.2f} ms")
        print(f"   Cost per Query:     ${best_model['cost_per_query_usd']:.6f}")
        print(f"   API Success Rate:   {best_model['api_success_rate']:.2f}%")
        print(f"   Memory Usage:       {best_model['memory_usage_mb']:.2f} MB")
        print(f"   GPU Utilization:    {best_model['gpu_utilization_percent']:.2f}%")
        
        print(f"\n⚙️  HYPERPARAMETERS:")
        for param, value in best_model['hyperparameters'].items():
            if value is not None:
                print(f"   {param:20} {value}")
        
        print("="*80 + "\n")
    
    def save_best_model_config(self, best_model: Dict[str, Any], output_path: str = "best_model_config.json"):
        """
        Save best model configuration to JSON file
        
        Args:
            best_model: Best model information
            output_path: Path to save configuration file
        """
        try:
            # Create config for deployment
            deployment_config = {
                "model_name": best_model["model_name"],
                "model_type": best_model["model_type"],
                "mlflow_run_id": best_model["run_id"],
                "performance_metrics": {
                    "composite_score": best_model["composite_score"],
                    "semantic_matching_score": best_model["semantic_matching_score"],
                    "hallucination_score": best_model["hallucination_score"],
                    "retrieval_score": best_model["retrieval_score"],
                    "api_success_rate": best_model["api_success_rate"]
                },
                "hyperparameters": best_model["hyperparameters"],
                "resource_requirements": {
                    "memory_mb": best_model["memory_usage_mb"],
                    "gpu_utilization_percent": best_model["gpu_utilization_percent"]
                },
                "cost_metrics": {
                    "cost_per_query_usd": best_model["cost_per_query_usd"],
                    "runtime_per_query_ms": best_model["runtime_per_query_ms"]
                }
            }
            
            with open(output_path, 'w') as f:
                json.dump(deployment_config, f, indent=2)
            
            logger.info(f"Best model config saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def compare_models_by_criteria(self, criteria: str = "cost") -> pd.DataFrame:
        """
        Compare models by specific criteria (cost, speed, accuracy)
        
        Args:
            criteria: One of 'cost', 'speed', 'accuracy', 'balanced'
            
        Returns:
            DataFrame with models sorted by criteria
        """
        if self.runs_df is None or self.runs_df.empty:
            return pd.DataFrame()
        
        df = self.runs_df.copy()
        
        # Add composite score if missing
        df = self.calculate_composite_scores(df)
        
        # Define sorting columns based on criteria
        sort_configs = {
            "cost": ("metrics.cost_per_query_usd", True),  # Lower is better
            "speed": ("metrics.runtime_per_query_ms", True),  # Lower is better
            "accuracy": ("metrics.composite_score", False),  # Higher is better
            "balanced": ("metrics.composite_score", False)  # Higher is better
        }
        
        if criteria not in sort_configs:
            logger.warning(f"Unknown criteria '{criteria}', using 'accuracy'")
            criteria = "accuracy"
        
        sort_col, ascending = sort_configs[criteria]
        
        # Select relevant columns
        cols_to_show = [
            "params.model_name",
            "metrics.composite_score",
            "metrics.semantic_matching_score",
            "metrics.hallucination_score",
            "metrics.runtime_per_query_ms",
            "metrics.cost_per_query_usd",
            "metrics.api_success_rate"
        ]
        
        available_cols = [col for col in cols_to_show if col in df.columns]
        comparison_df = df[available_cols].sort_values(sort_col, ascending=ascending)
        
        return comparison_df


def main():
    """Main function to select best model"""
    
    # Initialize selector
    selector = ModelSelector(experiment_name="RAG_Model_Selection")
    
    # Load experiment
    if not selector.load_experiment():
        logger.error("Failed to load experiment. Exiting.")
        return
    
    # Get all runs
    runs_df = selector.get_all_runs()
    if runs_df.empty:
        logger.error("No runs found. Exiting.")
        return
    
    # Select best model
    best_model = selector.select_best_model()
    if best_model is None:
        logger.error("Could not select best model. Exiting.")
        return
    
    # Get top 5 models
    top_models = selector.get_top_n_models(n=5)
    
    # Display results
    selector.display_results(best_model, top_models)
    
    # Save best model config
    selector.save_best_model_config(best_model)
    
    # Additional comparisons
    print("\n💰 TOP 3 BY COST:")
    print(selector.compare_models_by_criteria("cost").head(3).to_string(index=False))
    
    print("\n⚡ TOP 3 BY SPEED:")
    print(selector.compare_models_by_criteria("speed").head(3).to_string(index=False))
    
    print("\n" + "="*80)
    print("Model selection complete! Check 'best_model_config.json' for deployment config.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()