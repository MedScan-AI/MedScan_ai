"""
bias_detection_system.py - Comprehensive Bias Detection and Mitigation for Medical RAG
Implements slicing techniques, bias metrics tracking, and mitigation strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass, field
from pathlib import Path
import json
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Model and retrieval imports (assuming these exist in your project)
import sys
import os

# Add parent RAG directory to path to find sibling modules
current_dir = os.path.dirname(os.path.abspath(__file__))
rag_dir = os.path.dirname(current_dir)  # Go up to RAG/
sys.path.insert(0, rag_dir)

try:
    from models.models import ModelFactory
    from ModelInference.retrieval import DocumentRetriever, get_embeddings
    from ModelSelection.prompts import get_prompt_template
    from ModelInference.evaluate_rag import QAPair  # type: ignore
    from utils.config import source_mappings, COUNTRY_TLD_MAP
except ImportError as e:
    # These imports are only needed for the standalone main() function
    # The bias detection classes work without them
    print(f"Warning: Some imports failed: {e}")
    print("Bias detection will work, but standalone execution may not.")

    @dataclass
    class QAPair:
        """Fallback representation of a question/answer pair."""

        question: str
        answer: str
        metadata: Optional[Dict[str, Any]] = None

        @property
        def reference_answer(self) -> str:
            """Match the interface of the primary QAPair class."""
            return self.answer

        @classmethod
        def _extract_qa(cls, data: Dict[str, Any], idx: int) -> Optional["QAPair"]:
            """Create a QAPair from a dictionary with flexible keys."""
            question = (
                data.get("Q")
                or data.get("question")
                or data.get("query")
                or data.get("prompt")
            )
            answer = (
                data.get("A")
                or data.get("answer")
                or data.get("response")
                or data.get("completion")
            )

            if not question or not answer:
                print(
                    f"Warning: Skipping QA entry at index {idx} due to missing question/answer."
                )
                return None

            extra = {k: v for k, v in data.items() if k not in {"Q", "question", "query", "prompt", "A", "answer", "response", "completion"}}
            return cls(question=question, answer=answer, metadata=extra or None)

        @classmethod
        def from_dict(cls, data: Dict[str, Any], idx: int) -> Optional["QAPair"]:
            """Construct from a dictionary payload."""
            return cls._extract_qa(data, idx)

        @classmethod
        def from_line(cls, line: str, idx: int) -> Optional["QAPair"]:
            """Construct from a single line (JSON or delimiter-separated)."""
            cleaned = line.strip().rstrip(",")
            if not cleaned or cleaned in {"[", "]", "{", "}", "}"}:
                return None

            try:
                data = json.loads(cleaned)
            except json.JSONDecodeError:
                if "||" in cleaned:
                    question, answer = cleaned.split("||", 1)
                    return cls(question=question.strip(), answer=answer.strip())
                print(f"Warning: Could not parse QA line {idx}: {line}")
                return None

            return cls.from_dict(data, idx)

# Fairlearn imports for bias detection and mitigation
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    selection_rate,
    false_positive_rate,
    false_negative_rate,
    true_positive_rate,
    true_negative_rate
)
from fairlearn.reductions import (
    ExponentiatedGradient, 
    DemographicParity, 
    EqualizedOdds,
    BoundedGroupLoss
)
from fairlearn.postprocessing import ThresholdOptimizer

# Sklearn for additional metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class SliceMetrics:
    """Metrics for a specific data slice"""
    slice_name: str
    slice_value: str
    sample_size: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    false_negative_rate: float
    
    def to_dict(self) -> Dict:
        return {
            'slice_name': self.slice_name,
            'slice_value': self.slice_value,
            'sample_size': self.sample_size,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'fpr': self.false_positive_rate,
            'fnr': self.false_negative_rate
        }


@dataclass
class BiasAnalysisResult:
    """Comprehensive bias analysis results"""
    overall_metrics: Dict[str, float]
    slice_metrics: List[SliceMetrics]
    disparities: Dict[str, Dict[str, float]]
    fairness_violations: List[str]
    bias_detected: bool
    mitigation_recommendations: List[str]
    mitigation_applied: bool = False
    mitigated_metrics: Optional[Dict[str, float]] = None
    
    def to_json(self) -> str:
        return json.dumps({
            'overall_metrics': self.overall_metrics,
            'slice_metrics': [s.to_dict() for s in self.slice_metrics],
            'disparities': self.disparities,
            'fairness_violations': self.fairness_violations,
            'bias_detected': self.bias_detected,
            'mitigation_recommendations': self.mitigation_recommendations,
            'mitigation_applied': self.mitigation_applied,
            'mitigated_metrics': self.mitigated_metrics
        }, indent=2, default=str)


class ComprehensiveBiasDetector:
    """
    Comprehensive bias detection system with slicing techniques and mitigation strategies
    """
    
    def __init__(self, 
                 model,
                 bias_thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize bias detector
        
        Args:
            model: The ML model to analyze
            bias_thresholds: Thresholds for determining bias violations
        """
        self.model = model
        self.bias_thresholds = bias_thresholds or {
            'demographic_parity': 0.1,
            'equalized_odds': 0.1,
            'accuracy_disparity': 0.05,
            'f1_disparity': 0.05
        }
        self.results = None
        self.mitigation_models = {}
        
    def prepare_data(self,
                    queries: List[str],
                    contexts: List[str],
                    ground_truth: List[str],
                    metadata: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Generate predictions and prepare data for bias analysis
        """
        logger.info("Generating model predictions...")
        predictions = []
        prediction_scores = []
        
        prompt_template = get_prompt_template("prompt1")
        
        for query, context in zip(queries, contexts):
            prompt = prompt_template.replace("{context}", context).replace("{query}", query)
            result = self.model.infer(query=query, prompt=prompt)
            
            # Binary classification based on response quality
            if result["success"] and len(result["generated_text"]) > 20:
                pred = 1
                # Simulate confidence score
                score = min(0.5 + len(result["generated_text"]) / 200, 0.99)
            else:
                pred = 0
                score = 0.3
                
            predictions.append(pred)
            prediction_scores.append(score)
        
        # Convert ground truth to binary labels
        y_true = np.array([1 if len(gt) > 20 else 0 for gt in ground_truth])
        y_pred = np.array(predictions)
        y_scores = np.array(prediction_scores)
        
        # Add predictions to metadata
        metadata['y_true'] = y_true
        metadata['y_pred'] = y_pred
        metadata['y_scores'] = y_scores
        
        return y_true, y_pred, metadata
    
    def analyze_slices(self,
                      y_true: np.ndarray,
                      y_pred: np.ndarray,
                      metadata: pd.DataFrame,
                      slice_features: List[str]) -> Dict[str, List[SliceMetrics]]:
        """
        Perform detailed slice-based analysis
        """
        slice_results = {}
        
        for feature in slice_features:
            if feature not in metadata.columns:
                logger.warning(f"Feature {feature} not found in metadata")
                continue
                
            logger.info(f"Analyzing slices for feature: {feature}")
            feature_metrics = []
            
            # Get unique values for this feature
            unique_values = metadata[feature].unique()
            
            for value in unique_values:
                # Get indices for this slice
                slice_mask = metadata[feature] == value
                slice_y_true = y_true[slice_mask]
                slice_y_pred = y_pred[slice_mask]
                
                if len(slice_y_true) == 0:
                    continue
                
                # Calculate metrics for this slice
                metrics = self._calculate_slice_metrics(
                    slice_y_true, 
                    slice_y_pred,
                    feature,
                    str(value)
                )
                feature_metrics.append(metrics)
            
            slice_results[feature] = feature_metrics
        
        return slice_results
    
    def _calculate_slice_metrics(self,
                                y_true: np.ndarray,
                                y_pred: np.ndarray,
                                slice_name: str,
                                slice_value: str) -> SliceMetrics:
        """Calculate comprehensive metrics for a single slice"""
        
        # Handle edge cases
        if len(y_true) == 0:
            return SliceMetrics(
                slice_name=slice_name,
                slice_value=slice_value,
                sample_size=0,
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                false_positive_rate=0.0,
                false_negative_rate=0.0
            )
        
        # Calculate metrics with proper handling of edge cases
        accuracy = accuracy_score(y_true, y_pred)
        
        # Handle cases where we might not have positive/negative samples
        try:
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
        except:
            precision = recall = f1 = 0.0
        
        # Calculate FPR and FNR
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel() if len(np.unique(y_true)) > 1 else (0, 0, 0, len(y_true))
        
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        return SliceMetrics(
            slice_name=slice_name,
            slice_value=slice_value,
            sample_size=len(y_true),
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            false_positive_rate=fpr,
            false_negative_rate=fnr
        )
    
    def detect_disparities(self,
                          slice_results: Dict[str, List[SliceMetrics]]) -> Dict[str, Dict[str, float]]:
        """
        Detect disparities across slices
        """
        disparities = {}
        
        for feature, metrics_list in slice_results.items():
            if len(metrics_list) < 2:
                continue
                
            # Calculate disparities for each metric
            accuracies = [m.accuracy for m in metrics_list]
            f1_scores = [m.f1_score for m in metrics_list]
            fprs = [m.false_positive_rate for m in metrics_list]
            fnrs = [m.false_negative_rate for m in metrics_list]
            
            disparities[feature] = {
                'accuracy_range': max(accuracies) - min(accuracies),
                'accuracy_std': np.std(accuracies),
                'f1_range': max(f1_scores) - min(f1_scores),
                'f1_std': np.std(f1_scores),
                'fpr_range': max(fprs) - min(fprs),
                'fnr_range': max(fnrs) - min(fnrs),
                'worst_performing_slice': min(metrics_list, key=lambda x: x.accuracy).slice_value,
                'best_performing_slice': max(metrics_list, key=lambda x: x.accuracy).slice_value
            }
        
        return disparities
    
    def check_fairness_violations(self,
                                  disparities: Dict[str, Dict[str, float]]) -> List[str]:
        """
        Check for fairness violations based on thresholds
        """
        violations = []
        
        for feature, disparity_metrics in disparities.items():
            # Check accuracy disparity
            if disparity_metrics['accuracy_range'] > self.bias_thresholds['accuracy_disparity']:
                violations.append(
                    f"{feature}: Accuracy disparity ({disparity_metrics['accuracy_range']:.3f}) "
                    f"exceeds threshold ({self.bias_thresholds['accuracy_disparity']})"
                )
            
            # Check F1 disparity
            if disparity_metrics['f1_range'] > self.bias_thresholds['f1_disparity']:
                violations.append(
                    f"{feature}: F1 score disparity ({disparity_metrics['f1_range']:.3f}) "
                    f"exceeds threshold ({self.bias_thresholds['f1_disparity']})"
                )
        
        return violations
    
    def generate_mitigation_recommendations(self,
                                           disparities: Dict[str, Dict[str, float]],
                                           slice_results: Dict[str, List[SliceMetrics]]) -> List[str]:
        """
        Generate specific recommendations for bias mitigation
        """
        recommendations = []
        
        for feature, disparity_metrics in disparities.items():
            if disparity_metrics['accuracy_range'] > self.bias_thresholds['accuracy_disparity']:
                worst_slice = disparity_metrics['worst_performing_slice']
                best_slice = disparity_metrics['best_performing_slice']
                
                recommendations.append(
                    f"1. Re-sampling Strategy for {feature}:\n"
                    f"   - Oversample data from '{worst_slice}' group (current worst performer)\n"
                    f"   - Consider synthetic data generation for underrepresented cases"
                )
                
                recommendations.append(
                    f"2. Re-weighting Strategy for {feature}:\n"
                    f"   - Apply higher weights to '{worst_slice}' samples during training\n"
                    f"   - Suggested weight ratio: {1 + disparity_metrics['accuracy_range']:.2f}"
                )
                
                recommendations.append(
                    f"3. Threshold Optimization for {feature}:\n"
                    f"   - Adjust decision thresholds per group to equalize performance\n"
                    f"   - Consider using ThresholdOptimizer from Fairlearn"
                )
        
        if len(recommendations) == 0:
            recommendations.append("No significant bias detected. Model performs fairly across all slices.")
        
        return recommendations
    
    def apply_mitigation(self,
                        X: np.ndarray,
                        y_true: np.ndarray,
                        sensitive_features: pd.DataFrame,
                        mitigation_type: str = 'threshold') -> Dict[str, Any]:
        """
        Apply bias mitigation techniques
        """
        logger.info(f"Applying {mitigation_type} mitigation...")
        
        if mitigation_type == 'threshold':
            # Use ThresholdOptimizer for post-processing mitigation
            from sklearn.linear_model import LogisticRegression
            
            # Train a simple classifier for demonstration
            estimator = LogisticRegression(random_state=42, max_iter=1000)
            estimator.fit(X, y_true)
            
            # Apply threshold optimization
            postprocess = ThresholdOptimizer(
                estimator=estimator,
                constraints="demographic_parity",
                objective="accuracy_score"
            )
            
            postprocess.fit(X, y_true, sensitive_features=sensitive_features)
            y_pred_mitigated = postprocess.predict(X, sensitive_features=sensitive_features)
            
            # Calculate mitigated metrics
            mitigated_metrics = {
                'accuracy': accuracy_score(y_true, y_pred_mitigated),
                'demographic_parity_difference': demographic_parity_difference(
                    y_true, y_pred_mitigated, sensitive_features=sensitive_features
                )
            }
            
            return {
                'mitigated_predictions': y_pred_mitigated,
                'mitigated_metrics': mitigated_metrics,
                'mitigation_model': postprocess
            }
        
        elif mitigation_type == 'reweighting':
            # Implement sample reweighting
            logger.info("Reweighting strategy selected - would require model retraining")
            return {
                'recommendation': 'Retrain model with weighted samples',
                'weights': self._calculate_sample_weights(y_true, sensitive_features)
            }
        
        else:
            logger.warning(f"Unknown mitigation type: {mitigation_type}")
            return {}
    
    def _calculate_sample_weights(self,
                                 y_true: np.ndarray,
                                 sensitive_features: pd.DataFrame) -> np.ndarray:
        """Calculate sample weights for reweighting strategy"""
        weights = np.ones(len(y_true))
        
        for col in sensitive_features.columns:
            unique_values = sensitive_features[col].unique()
            value_counts = sensitive_features[col].value_counts()
            
            for value in unique_values:
                mask = sensitive_features[col] == value
                # Inverse frequency weighting
                weights[mask] *= len(sensitive_features) / (len(unique_values) * value_counts[value])
        
        return weights / weights.mean()
    
    def visualize_bias_analysis(self,
                               slice_results: Dict[str, List[SliceMetrics]],
                               save_path: Optional[str] = None):
        """
        Create comprehensive visualization of bias analysis
        """
        num_features = len(slice_results)
        fig, axes = plt.subplots(2, num_features, figsize=(6*num_features, 10))
        
        if num_features == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, (feature, metrics_list) in enumerate(slice_results.items()):
            # Sort metrics by accuracy for better visualization
            metrics_list = sorted(metrics_list, key=lambda x: x.accuracy)
            
            # Extract data for plotting
            slice_names = [m.slice_value for m in metrics_list]
            accuracies = [m.accuracy for m in metrics_list]
            f1_scores = [m.f1_score for m in metrics_list]
            sample_sizes = [m.sample_size for m in metrics_list]
            
            # Plot 1: Accuracy by slice
            ax1 = axes[0, idx]
            bars1 = ax1.bar(range(len(slice_names)), accuracies, color='steelblue')
            
            # Highlight worst performer
            min_acc_idx = accuracies.index(min(accuracies))
            bars1[min_acc_idx].set_color('red')
            
            # Highlight best performer
            max_acc_idx = accuracies.index(max(accuracies))
            bars1[max_acc_idx].set_color('green')
            
            ax1.set_xlabel('Slice')
            ax1.set_ylabel('Accuracy')
            ax1.set_title(f'Accuracy by {feature}')
            ax1.set_xticks(range(len(slice_names)))
            ax1.set_xticklabels(slice_names, rotation=45, ha='right')
            ax1.grid(True, alpha=0.3)
            
            # Add sample size annotations
            for i, (acc, size) in enumerate(zip(accuracies, sample_sizes)):
                ax1.text(i, acc + 0.01, f'n={size}', ha='center', fontsize=8)
            
            # Plot 2: F1 Score by slice
            ax2 = axes[1, idx]
            bars2 = ax2.bar(range(len(slice_names)), f1_scores, color='coral')
            
            # Highlight worst and best
            if f1_scores:
                min_f1_idx = f1_scores.index(min(f1_scores))
                bars2[min_f1_idx].set_color('darkred')
                max_f1_idx = f1_scores.index(max(f1_scores))
                bars2[max_f1_idx].set_color('darkgreen')
            
            ax2.set_xlabel('Slice')
            ax2.set_ylabel('F1 Score')
            ax2.set_title(f'F1 Score by {feature}')
            ax2.set_xticks(range(len(slice_names)))
            ax2.set_xticklabels(slice_names, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Bias Analysis: Performance Disparities Across Slices', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def generate_detailed_report(self, result: BiasAnalysisResult) -> str:
        """
        Generate a comprehensive bias analysis report
        """
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE BIAS DETECTION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Overall metrics
        report.append("OVERALL MODEL PERFORMANCE:")
        report.append("-" * 40)
        for metric, value in result.overall_metrics.items():
            report.append(f"  {metric}: {value:.4f}")
        report.append("")
        
        # Slice performance summary
        report.append("SLICE-BASED PERFORMANCE ANALYSIS:")
        report.append("-" * 40)
        
        # Group slices by feature
        slices_by_feature = {}
        for slice_metric in result.slice_metrics:
            if slice_metric.slice_name not in slices_by_feature:
                slices_by_feature[slice_metric.slice_name] = []
            slices_by_feature[slice_metric.slice_name].append(slice_metric)
        
        for feature, slices in slices_by_feature.items():
            report.append(f"\n{feature.upper()}:")
            report.append("  " + "-" * 36)
            
            # Sort by accuracy for clarity
            slices = sorted(slices, key=lambda x: x.accuracy, reverse=True)
            
            for slice_metric in slices:
                report.append(f"  {slice_metric.slice_value}:")
                report.append(f"    Sample Size: {slice_metric.sample_size}")
                report.append(f"    Accuracy: {slice_metric.accuracy:.3f}")
                report.append(f"    F1 Score: {slice_metric.f1_score:.3f}")
                report.append(f"    Precision: {slice_metric.precision:.3f}")
                report.append(f"    Recall: {slice_metric.recall:.3f}")
                report.append(f"    FPR: {slice_metric.false_positive_rate:.3f}")
                report.append(f"    FNR: {slice_metric.false_negative_rate:.3f}")
                report.append("")
        
        # Disparities analysis
        report.append("DISPARITY ANALYSIS:")
        report.append("-" * 40)
        for feature, disparities in result.disparities.items():
            report.append(f"\n{feature}:")
            for metric, value in disparities.items():
                if isinstance(value, float):
                    report.append(f"  {metric}: {value:.4f}")
                else:
                    report.append(f"  {metric}: {value}")
        report.append("")
        
        # Bias detection results
        report.append("BIAS DETECTION:")
        report.append("-" * 40)
        report.append(f"Bias Detected: {'YES' if result.bias_detected else 'NO'}")
        
        if result.fairness_violations:
            report.append("\nFairness Violations:")
            for violation in result.fairness_violations:
                report.append(f"  - {violation}")
        else:
            report.append("No fairness violations detected")
        report.append("")
        
        # Mitigation recommendations
        report.append("MITIGATION RECOMMENDATIONS:")
        report.append("-" * 40)
        for i, rec in enumerate(result.mitigation_recommendations, 1):
            report.append(f"\n{rec}")
        report.append("")
        
        # Mitigation results (if applied)
        if result.mitigation_applied and result.mitigated_metrics:
            report.append("MITIGATION RESULTS:")
            report.append("-" * 40)
            for metric, value in result.mitigated_metrics.items():
                report.append(f"  {metric}: {value:.4f}")
            report.append("")
        
        report.append("=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def run_comprehensive_analysis(self,
                                  y_true: np.ndarray,
                                  y_pred: np.ndarray,
                                  metadata: pd.DataFrame,
                                  slice_features: List[str],
                                  apply_mitigation: bool = False) -> BiasAnalysisResult:
        """
        Run comprehensive bias analysis pipeline
        """
        logger.info("Starting comprehensive bias analysis...")
        
        # Calculate overall metrics
        overall_metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        
        # Analyze slices
        slice_results = self.analyze_slices(y_true, y_pred, metadata, slice_features)
        
        # Flatten slice results for the result object
        all_slice_metrics = []
        for feature_slices in slice_results.values():
            all_slice_metrics.extend(feature_slices)
        
        # Detect disparities
        disparities = self.detect_disparities(slice_results)
        
        # Check for fairness violations
        violations = self.check_fairness_violations(disparities)
        
        # Generate recommendations
        recommendations = self.generate_mitigation_recommendations(disparities, slice_results)
        
        # Create result object
        result = BiasAnalysisResult(
            overall_metrics=overall_metrics,
            slice_metrics=all_slice_metrics,
            disparities=disparities,
            fairness_violations=violations,
            bias_detected=len(violations) > 0,
            mitigation_recommendations=recommendations
        )
        
        # Apply mitigation if requested
        if apply_mitigation and result.bias_detected:
            logger.info("Applying bias mitigation strategies...")
            
            # Create feature matrix for mitigation
            X = metadata[['y_scores']].values
            
            # Apply mitigation for each sensitive feature
            for feature in slice_features:
                if feature in metadata.columns:
                    mitigation_result = self.apply_mitigation(
                        X, y_true, 
                        metadata[[feature]], 
                        mitigation_type='threshold'
                    )
                    
                    if 'mitigated_metrics' in mitigation_result:
                        result.mitigation_applied = True
                        result.mitigated_metrics = mitigation_result['mitigated_metrics']
                        break  # Apply mitigation for the first feature with violations
        
        # Store results
        self.results = result
        
        # Visualize results
        self.visualize_bias_analysis(slice_results)
        
        return result


def main():
    """
    Main execution function for bias detection analysis
    """
    logger.info("Starting Bias Detection Analysis for Medical RAG System")
    logger.info("=" * 80)
    
    # Configuration
    config = {
        'qa_file': 'data/qa.txt',
        'index_path': 'data/index.bin',
        'chunks_path': 'data/data.pkl',
        'model_name': 'flan_t5',
        'output_dir': 'bias_analysis_results',
        'slice_features': [
            'query_complexity',
            'medical_specialty',
            'source_class',
            'country_class',
            'publish_year_category'
        ],
        'apply_mitigation': True
    }
    
    # Create output directory
    output_path = Path(config['output_dir'])
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Load QA pairs
    logger.info("Loading QA pairs...")
    qa_pairs = load_qa_pairs(config['qa_file'])
    queries = [pair.question for pair in qa_pairs]
    ground_truths = [pair.reference_answer for pair in qa_pairs]
    
    # Retrieve contexts
    logger.info("Retrieving contexts...")
    contexts = retrieve_contexts(
        queries,
        config['index_path'],
        config['chunks_path']
    )
    
    # Create metadata
    logger.info("Creating metadata for bias analysis...")
    metadata = create_comprehensive_metadata(
        queries,
        ground_truths,
        contexts,
        source_mappings,
        COUNTRY_TLD_MAP
    )
    
    # Initialize model
    logger.info(f"Initializing model: {config['model_name']}")
    model = ModelFactory.create_model(config['model_name'])
    
    # Initialize bias detector
    detector = ComprehensiveBiasDetector(
        model=model,
        bias_thresholds={
            'accuracy_disparity': 0.05,
            'f1_disparity': 0.05,
            'demographic_parity': 0.1,
            'equalized_odds': 0.1
        }
    )
    
    # Prepare data and generate predictions
    y_true, y_pred, metadata = detector.prepare_data(
        queries, contexts, ground_truths, metadata
    )
    
    # Run comprehensive analysis
    result = detector.run_comprehensive_analysis(
        y_true=y_true,
        y_pred=y_pred,
        metadata=metadata,
        slice_features=config['slice_features'],
        apply_mitigation=config['apply_mitigation']
    )
    
    # Generate and save report
    report = detector.generate_detailed_report(result)
    report_path = output_path / 'comprehensive_bias_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"Report saved to {report_path}")
    
    # Save results as JSON
    json_path = output_path / 'bias_analysis_results.json'
    with open(json_path, 'w') as f:
        f.write(result.to_json())
    logger.info(f"Results saved to {json_path}")
    
    # Save visualizations
    viz_path = output_path / 'bias_visualizations.png'
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    logger.info(f"Visualizations saved to {viz_path}")
    
    logger.info("=" * 80)
    logger.info("Bias Detection Analysis Complete")
    logger.info(f"All results saved to: {output_path}")
    
    return result


def load_qa_pairs(path: str) -> List[QAPair]:
    """Load Q and A pairs from a JSON file or newline-delimited records."""
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"QA file not found at {path}")

    raw_text = path_obj.read_text(encoding="utf8").strip()
    qa_pairs: List[QAPair] = []

    if not raw_text:
        return qa_pairs

    # Primary path: treat as JSON (list or single object)
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, list):
        for idx, item in enumerate(payload, start=1):
            if hasattr(QAPair, "from_dict") and callable(getattr(QAPair, "from_dict")):
                qa = QAPair.from_dict(item, idx)  # type: ignore[attr-defined]
            else:
                try:
                    qa = QAPair(**item)  # type: ignore[arg-type]
                except TypeError:
                    qa = None
            if qa is not None:
                qa_pairs.append(qa)
        return qa_pairs

    if isinstance(payload, dict):
        if hasattr(QAPair, "from_dict") and callable(getattr(QAPair, "from_dict")):
            qa = QAPair.from_dict(payload, 1)  # type: ignore[attr-defined]
        else:
            try:
                qa = QAPair(**payload)  # type: ignore[arg-type]
            except TypeError:
                qa = None
        if qa is not None:
            qa_pairs.append(qa)
        return qa_pairs

    # Fallback: treat file as newline-delimited JSON/records
    current_record = ""
    brace_depth = 0

    for idx, line in enumerate(raw_text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue

        if stripped in {"[", "]"}:
            continue

        brace_depth += stripped.count("{") - stripped.count("}")
        current_record += stripped

        # Completed JSON object
        if brace_depth <= 0 and current_record:
            candidate = current_record.rstrip(",")
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError:
                if hasattr(QAPair, "from_line") and callable(getattr(QAPair, "from_line")):
                    qa = QAPair.from_line(candidate, idx)  # type: ignore[attr-defined]
                else:
                    qa = None
            else:
                if hasattr(QAPair, "from_dict") and callable(getattr(QAPair, "from_dict")):
                    qa = QAPair.from_dict(data, idx)  # type: ignore[attr-defined]
                else:
                    try:
                        qa = QAPair(**data)  # type: ignore[arg-type]
                    except TypeError:
                        qa = None

            if qa is not None:
                qa_pairs.append(qa)

            current_record = ""
            brace_depth = 0

    return qa_pairs


def retrieve_contexts(queries: List[str],
                     index_path: str,
                     chunks_path: str,
                     num_docs: int = 5) -> List[str]:
    """Retrieve contexts for queries"""
    retriever = DocumentRetriever(
        index_path=Path(index_path),
        chunks_path=Path(chunks_path)
    )
    
    contexts = []
    for query in queries:
        query_emb = get_embeddings(query)
        docs = retriever.retrieve(query_embedding=query_emb, num_docs=num_docs)
        context = retriever.format_context_for_prompt(docs)
        contexts.append(context)
    
    return contexts


def create_comprehensive_metadata(queries: List[str],
                                 ground_truths: List[str],
                                 contexts: List[str],
                                 source_mappings: Dict[str, List[str]],
                                 country_tld_map: Dict[str, str]) -> pd.DataFrame:
    """Create comprehensive metadata for bias analysis"""
    metadata = []
    
    for idx, (query, truth, context) in enumerate(zip(queries, ground_truths, contexts)):
        q_lower = query.lower()
        t_lower = truth.lower()
        combined = q_lower + " " + t_lower
        
        # Query complexity
        word_count = len(query.split())
        if word_count > 15 or any(term in q_lower for term in ["mechanism", "pathophysiology", "differential"]):
            complexity = "complex"
        elif word_count > 8:
            complexity = "moderate"
        else:
            complexity = "simple"
        
        # Medical specialty
        if any(term in combined for term in ["tb", "tuberculosis", "mycobacterium"]):
            specialty = "respiratory"
        elif any(term in combined for term in ["lung cancer", "nsclc", "sclc"]):
            specialty = "oncology_lung"
        elif any(term in combined for term in ["cancer", "chemotherapy", "radiation"]):
            specialty = "oncology_general"
        else:
            specialty = "general_medicine"
        
        # Extract source class
        source_value = "unknown"
        lower_context = context.lower()
        for group, patterns in source_mappings.items():
            if any(p.lower() in lower_context for p in patterns):
                source_value = group
                break
        
        # Extract country class
        country_value = "unknown"
        for tld, cname in country_tld_map.items():
            if f".{tld}" in lower_context:
                country_value = cname
                break
        
        # Extract and categorize publish year
        year_match = re.search(r"(20|19)\d{2}", context)
        if year_match:
            year = int(year_match.group(0))
            if year >= 2020:
                year_category = "recent"
            elif year >= 2010:
                year_category = "moderate"
            else:
                year_category = "old"
        else:
            year_category = "unknown"
        
        metadata.append({
            "query_complexity": complexity,
            "medical_specialty": specialty,
            "source_class": source_value,
            "country_class": country_value,
            "publish_year_category": year_category
        })
    
    return pd.DataFrame(metadata)


if __name__ == "__main__":
    result = main()