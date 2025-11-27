"""
Metrics tracking and evaluation utilities.
Tracks comprehensive metrics (accuracy, precision, recall, F1-score, AUC) across epochs.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from datetime import datetime
import logging
import warnings

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.exceptions import UndefinedMetricWarning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

logger = logging.getLogger(__name__)


class MetricsTracker(keras.callbacks.Callback):
    """Callback to track comprehensive metrics during training."""
    
    def __init__(
        self,
        val_generator: ImageDataGenerator,
        class_names: List[str],
        output_dir: Path,
        model_name: str,
        dataset_name: str,
        model: Optional[keras.Model] = None
    ):
        """
        Initialize MetricsTracker.
        
        Args:
            val_generator: Validation data generator
            class_names: List of class names
            output_dir: Directory to save metrics
            model_name: Name of the model
            dataset_name: Name of the dataset
            model: Optional model reference (for hyperparameter tracking)
        """
        super().__init__()
        self.val_generator = val_generator
        self.class_names = class_names
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.num_classes = len(class_names)
        self.model_ref = model
        
        # Store metrics per epoch
        self.epoch_metrics = []
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Compute and store comprehensive metrics at end of each epoch."""
        logs = logs or {}
        
        # Get validation data
        try:
            # Reset generator
            self.val_generator.reset()
            
            # Collect predictions and true labels
            y_true = []
            y_pred_proba = []
            
            # Calculate steps
            steps = None
            if hasattr(self.val_generator, '_steps_per_epoch'):
                steps = self.val_generator._steps_per_epoch
            elif hasattr(self.val_generator, 'samples') and self.val_generator.samples > 0:
                steps = (self.val_generator.samples + self.val_generator.batch_size - 1) // self.val_generator.batch_size
            
            # Get predictions
            for batch_idx in range(steps if steps else len(self.val_generator)):
                try:
                    batch = next(self.val_generator)
                    if len(batch) == 2:
                        x_batch, y_batch = batch
                    else:
                        continue
                    
                    # Get predictions
                    pred_proba = self.model.predict(x_batch, verbose=0)
                    y_pred_proba.append(pred_proba)
                    y_true.append(y_batch)
                    
                    if steps and batch_idx >= steps - 1:
                        break
                except StopIteration:
                    break
                except Exception as e:
                    logger.warning(f"Error processing batch {batch_idx}: {e}")
                    continue
            
            if len(y_true) == 0:
                logger.warning(f"No validation data collected at epoch {epoch + 1}")
                return
            
            # Concatenate all batches
            y_true = np.concatenate(y_true, axis=0)
            y_pred_proba = np.concatenate(y_pred_proba, axis=0)
            
            # Get predicted classes
            y_pred = np.argmax(y_pred_proba, axis=1)
            y_true_classes = np.argmax(y_true, axis=1)
            
            # Compute metrics
            metrics = self._compute_metrics(y_true_classes, y_pred, y_pred_proba)
            
            # Extract hyperparameters from model if available
            hyperparameters = {}
            model_to_check = self.model if hasattr(self, 'model') and self.model else self.model_ref
            if model_to_check:
                try:
                    # Get learning rate from optimizer
                    if hasattr(model_to_check, 'optimizer') and model_to_check.optimizer:
                        lr_var = model_to_check.optimizer.learning_rate
                        if hasattr(lr_var, 'numpy'):
                            lr = float(lr_var.numpy())
                        elif hasattr(lr_var, 'value'):
                            lr = float(lr_var.value())
                        else:
                            lr = float(lr_var)
                        hyperparameters['learning_rate'] = lr
                except Exception as e:
                    logger.debug(f"Could not extract learning rate: {e}")
                    pass
            
            # Add epoch number, training metrics, and hyperparameters from logs
            epoch_metric = {
                'epoch': epoch + 1,
                'train_loss': logs.get('loss', 0.0),
                'train_accuracy': logs.get('accuracy', 0.0),
                'val_loss': logs.get('val_loss', 0.0),
                'val_accuracy': logs.get('val_accuracy', 0.0),
                **metrics,
                **hyperparameters
            }
            
            self.epoch_metrics.append(epoch_metric)
            
            # Save metrics to file after each epoch
            self._save_metrics()
            
        except Exception as e:
            logger.error(f"Error computing metrics at epoch {epoch + 1}: {e}", exc_info=True)
    
    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """Compute comprehensive metrics."""
        metrics = {}
        
        try:
            # Basic metrics (suppress warnings by using zero_division=0)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
                metrics['precision'] = float(precision_score(
                    y_true, y_pred, average='weighted', zero_division=0
                ))
                metrics['recall'] = float(recall_score(
                    y_true, y_pred, average='weighted', zero_division=0
                ))
                metrics['f1_score'] = float(f1_score(
                    y_true, y_pred, average='weighted', zero_division=0
                ))
                metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
                
                # Per-class metrics
                metrics['per_class_precision'] = precision_score(
                    y_true, y_pred, average=None, zero_division=0
                ).tolist()
                metrics['per_class_recall'] = recall_score(
                    y_true, y_pred, average=None, zero_division=0
                ).tolist()
                metrics['per_class_f1'] = f1_score(
                    y_true, y_pred, average=None, zero_division=0
                ).tolist()
            
            # AUC (handle binary and multiclass)
            try:
                if self.num_classes == 2:
                    # Binary classification
                    metrics['auc'] = float(roc_auc_score(
                        y_true, y_pred_proba[:, 1]
                    ))
                else:
                    # Multiclass - use one-vs-rest
                    metrics['auc'] = float(roc_auc_score(
                        y_true, y_pred_proba, multi_class='ovr', average='weighted'
                    ))
            except Exception as e:
                logger.warning(f"Could not compute AUC: {e}")
                metrics['auc'] = 0.0
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            
            # Extract support (number of samples per class) from classification report
            try:
                report = classification_report(
                    y_true, y_pred, target_names=self.class_names, output_dict=True, zero_division=0
                )
                # Extract support per class
                per_class_support = []
                for class_name in self.class_names:
                    if class_name in report:
                        per_class_support.append(float(report[class_name].get('support', 0)))
                metrics['per_class_support'] = per_class_support
                metrics['classification_report'] = report
            except Exception as e:
                logger.warning(f"Could not compute classification report: {e}")
                # Calculate support from confusion matrix
                per_class_support = [float(cm[i, :].sum()) for i in range(len(self.class_names))]
                metrics['per_class_support'] = per_class_support
            
        except Exception as e:
            logger.error(f"Error computing metrics: {e}", exc_info=True)
            # Return default metrics
            metrics = {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'accuracy': 0.0,
                'auc': 0.0
            }
        
        return metrics
    
    def _save_metrics(self):
        """Save metrics to JSON file."""
        metrics_file = self.output_dir / f"{self.model_name}_epoch_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump({
                'model_name': self.model_name,
                'dataset_name': self.dataset_name,
                'class_names': self.class_names,
                'num_classes': self.num_classes,
                'epoch_metrics': self.epoch_metrics,
                'last_updated': datetime.now().isoformat()
            }, f, indent=2, default=str)
    
    def get_final_metrics(self) -> Dict:
        """Get final metrics from last epoch."""
        if self.epoch_metrics:
            return self.epoch_metrics[-1]
        return {}


def compute_comprehensive_metrics(
    model: keras.Model,
    test_generator: ImageDataGenerator,
    class_names: List[str]
) -> Dict[str, any]:
    """
    Compute comprehensive metrics on test set.
    
    Args:
        model: Trained Keras model
        test_generator: Test data generator
        class_names: List of class names
        
    Returns:
        Dictionary of metrics including confusion matrix
    """
    logger.info("Computing comprehensive test metrics...")
    
    # Collect predictions and true labels
    y_true = []
    y_pred_proba = []
    
    # Calculate steps
    steps = None
    if hasattr(test_generator, '_steps_per_epoch'):
        steps = test_generator._steps_per_epoch
    elif hasattr(test_generator, 'samples') and test_generator.samples > 0:
        steps = (test_generator.samples + test_generator.batch_size - 1) // test_generator.batch_size
    
    # Reset generator
    test_generator.reset()
    
    # Get predictions
    for batch_idx in range(steps if steps else len(test_generator)):
        try:
            batch = next(test_generator)
            if len(batch) == 2:
                x_batch, y_batch = batch
            else:
                continue
            
            # Get predictions
            pred_proba = model.predict(x_batch, verbose=0)
            y_pred_proba.append(pred_proba)
            y_true.append(y_batch)
            
            if steps and batch_idx >= steps - 1:
                break
        except StopIteration:
            break
        except Exception as e:
            logger.warning(f"Error processing batch {batch_idx}: {e}")
            continue
    
    if len(y_true) == 0:
        logger.warning("No test data collected")
        return {}
    
    # Concatenate all batches
    y_true = np.concatenate(y_true, axis=0)
    y_pred_proba = np.concatenate(y_pred_proba, axis=0)
    
    # Get predicted classes
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true_classes = np.argmax(y_true, axis=1)
    
    num_classes = len(class_names)
    
    # Compute metrics (suppress warnings by using zero_division=0 and warning filter)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
        metrics = {
            'accuracy': float(accuracy_score(y_true_classes, y_pred)),
            'precision': float(precision_score(
                y_true_classes, y_pred, average='weighted', zero_division=0
            )),
            'recall': float(recall_score(
                y_true_classes, y_pred, average='weighted', zero_division=0
            )),
            'f1_score': float(f1_score(
                y_true_classes, y_pred, average='weighted', zero_division=0
            ))
        }
        
        # Per-class metrics
        metrics['per_class_precision'] = precision_score(
            y_true_classes, y_pred, average=None, zero_division=0
        ).tolist()
        metrics['per_class_recall'] = recall_score(
            y_true_classes, y_pred, average=None, zero_division=0
        ).tolist()
        metrics['per_class_f1'] = f1_score(
            y_true_classes, y_pred, average=None, zero_division=0
        ).tolist()
        
        # Classification report
        report = classification_report(
            y_true_classes, y_pred, target_names=class_names, output_dict=True, zero_division=0
        )
        metrics['classification_report'] = report
    
    # AUC
    try:
        if num_classes == 2:
            metrics['auc'] = float(roc_auc_score(
                y_true_classes, y_pred_proba[:, 1]
            ))
        else:
            metrics['auc'] = float(roc_auc_score(
                y_true_classes, y_pred_proba, multi_class='ovr', average='weighted'
            ))
    except Exception as e:
        logger.warning(f"Could not compute AUC: {e}")
        metrics['auc'] = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics

