"""
RAG Bias Adapter - Connects bias detector with  pipeline
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
import logging
import mlflow

# Import bias detector 
from rag_bias_check import ComprehensiveBiasDetector

logger = logging.getLogger(__name__)


class RAGBiasAdapter:
    """Adapts continuous RAG scores to binary labels for bias detection"""
    
    def __init__(self, semantic_threshold: float = 0.5, hallucination_threshold: float = 0.7):
        self.semantic_threshold = semantic_threshold
        self.hallucination_threshold = hallucination_threshold
        
        # Initialize detector
        self.detector = ComprehensiveBiasDetector(
            model=None,
            bias_thresholds={
                'accuracy_disparity': 0.10,
                'f1_disparity': 0.10
            }
        )
    
    def create_metadata_from_queries(
        self, 
        queries: List[str],
        semantic_scores: List[float],
        hallucination_scores: List[float]
    ) -> pd.DataFrame:
        """Create metadata with RAG-specific slices"""
        
        metadata_records = []
        
        for query, sem_score, hall_score in zip(queries, semantic_scores, hallucination_scores):
            q_lower = query.lower()
            
            # Query complexity
            word_count = len(query.split())
            if word_count > 20:
                complexity = "complex"
            elif word_count > 10:
                complexity = "moderate"
            else:
                complexity = "simple"
            
            # Medical domain
            if any(term in q_lower for term in ["tb", "tuberculosis"]):
                domain = "tuberculosis"
            elif any(term in q_lower for term in ["lung cancer", "nsclc"]):
                domain = "lung_cancer"
            else:
                domain = "general"
            
            # Query type
            if any(term in q_lower for term in ["symptom", "sign"]):
                query_type = "symptom"
            elif any(term in q_lower for term in ["diagnos", "test"]):
                query_type = "diagnosis"
            elif any(term in q_lower for term in ["treat", "therap"]):
                query_type = "treatment"
            else:
                query_type = "general"
            
            metadata_records.append({
                "query_complexity": complexity,
                "medical_domain": domain,
                "query_type": query_type,
                "semantic_score": sem_score,
                "hallucination_score": hall_score
            })
        
        return pd.DataFrame(metadata_records)
    
    def convert_to_binary_labels(
        self, 
        per_query_results: List[Dict[str, Any]]
    ) -> tuple:
        """Convert RAG continuous scores to binary labels"""
        
        # Extract data
        queries = []
        semantic_scores = []
        hallucination_scores = []
        
        for result in per_query_results:
            if "error" in result:
                continue
            queries.append(result.get("query", ""))
            semantic_scores.append(result.get("semantic_matching_score", 0.0))
            hallucination_scores.append(result.get("hallucination_score", 0.0))
        
        # Convert to binary
        combined_scores = np.array([
            0.5 * sem + 0.5 * hall 
            for sem, hall in zip(semantic_scores, hallucination_scores)
        ])
        
        y_true = (combined_scores > 0.6).astype(int)
        y_pred = (
            (np.array(semantic_scores) > self.semantic_threshold) & 
            (np.array(hallucination_scores) > self.hallucination_threshold)
        ).astype(int)
        
        metadata = self.create_metadata_from_queries(
            queries, semantic_scores, hallucination_scores
        )
        
        return y_true, y_pred, metadata
    
    def run_bias_analysis(
        self,
        per_query_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run bias analysis on RAG results"""
        
        logger.info("Starting bias analysis...")
        
        # Convert to binary labels
        y_true, y_pred, metadata = self.convert_to_binary_labels(per_query_results)
        
        # Run analysis
        result = self.detector.run_comprehensive_analysis(
            y_true=y_true,
            y_pred=y_pred,
            metadata=metadata,
            slice_features=["query_complexity", "medical_domain", "query_type"],
            apply_mitigation=False
        )
        
        # Convert to dict for MLflow
        analysis_dict = {
            "bias_detected": result.bias_detected,
            "overall_accuracy": result.overall_metrics.get("accuracy", 0.0),
            "overall_f1": result.overall_metrics.get("f1_score", 0.0),
            "num_violations": len(result.fairness_violations),
            "violations": result.fairness_violations,
            "disparities": result.disparities,
            "recommendations": result.mitigation_recommendations
        }
        
        logger.info(f"Bias detected: {analysis_dict['bias_detected']}")
        logger.info(f"Violations: {analysis_dict['num_violations']}")
        
        return analysis_dict
    
    def log_to_mlflow(self, results: Dict[str, Any]):
        """Log bias results to MLflow"""
        
        mlflow.log_metric("bias_detected", int(results["bias_detected"]))
        mlflow.log_metric("bias_violations_count", results["num_violations"])
        mlflow.log_metric("bias_overall_accuracy", results["overall_accuracy"])
        mlflow.log_metric("bias_overall_f1", results["overall_f1"])
        
        # Log violations
        if results["violations"]:
            mlflow.log_dict(
                {"violations": results["violations"]},
                "bias_violations.json"
            )
        
        # Log recommendations
        mlflow.log_dict(
            {"recommendations": results["recommendations"]},
            "bias_recommendations.json"
        )
        
        logger.info("Bias results logged to MLflow")


def run_bias_check(best_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simple function to run bias check and log to MLflow.
    Call this after HPO completes.
    """
    
    bias_adapter = RAGBiasAdapter(
        semantic_threshold=0.5,
        hallucination_threshold=0.7
    )
    
    bias_results = bias_adapter.run_bias_analysis(
        per_query_results=best_metrics["per_query_results"]
    )
    
    bias_adapter.log_to_mlflow(bias_results)
    
    # Print summary
    print("\n" + "="*60)
    print("BIAS ANALYSIS")
    print("="*60)
    print(f"Bias Detected: {bias_results['bias_detected']}")
    print(f"Violations: {bias_results['num_violations']}")
    
    if bias_results['bias_detected']:
        print("\nViolations:")
        for v in bias_results['violations']:
            print(f"  - {v}")
    
    print("="*60 + "\n")
    
    return bias_results