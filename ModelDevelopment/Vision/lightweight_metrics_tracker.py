"""
Lightweight metrics tracking callback for hyperparameter tuning.
This version doesn't store non-picklable objects, making it compatible with KerasTuner.
"""

import numpy as np
import json
from typing import Dict, List, Optional
from pathlib import Path
import logging
import warnings

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report
)
from sklearn.exceptions import UndefinedMetricWarning
import tensorflow as tf
from tensorflow import keras

logger = logging.getLogger(__name__)


class LightweightMetricsTracker(keras.callbacks.Callback):
    """
    Lightweight callback to track metrics during hyperparameter tuning.
    This version stores only picklable data (paths, not generators).
    """
    
    def __init__(
        self,
        val_data_path: str,
        class_names: List[str],
        output_file: Path,
        num_classes: int,
        use_validation_subset: bool = False,
        validation_split: float = 0.2
    ):
        """
        Initialize LightweightMetricsTracker.
        
        Args:
            val_data_path: Path to validation data directory (or train directory if use_validation_subset=True)
            class_names: List of class names
            output_file: Path to JSON file to save metrics
            num_classes: Number of classes
            use_validation_subset: If True, validation is a subset of training data
            validation_split: Fraction of data to use for validation (if use_validation_subset=True)
        """
        super().__init__()
        self.val_data_path = val_data_path  # Store path, not generator
        self.class_names = class_names
        self.output_file = Path(output_file)
        self.num_classes = num_classes
        self.use_validation_subset = use_validation_subset
        self.validation_split = validation_split
        
        # Store metrics per epoch
        self.epoch_metrics = []
        
        # Track current trial number (for hyperparameter tuning)
        self.current_trial = None
        
        # Create output directory
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
    
    def on_train_begin(self, logs=None):
        """Called at the beginning of training (each trial)."""
        # Determine trial number by checking existing metrics in the file
        # Each new training run (trial) will have epoch 0, so we count existing trials
        try:
            if self.output_file.exists():
                with open(self.output_file, 'r') as f:
                    existing_data = json.load(f)
                    existing_metrics = existing_data.get('epoch_metrics', [])
                    if existing_metrics:
                        # Get the highest trial number and increment
                        max_trial = max(m.get('trial', 1) for m in existing_metrics)
                        self.current_trial = max_trial + 1
                    else:
                        self.current_trial = 1
            else:
                # First trial
                self.current_trial = 1
        except Exception as e:
            logger.debug(f"Could not determine trial from file: {e}")
            self.current_trial = 1
        
        # Initialize epoch tracking
        self._last_epoch = -1
        
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Compute and store comprehensive metrics at end of each epoch."""
        try:
            logs = logs or {}
            
            # Trial number is set in on_train_begin, so we don't need to detect it here
            # Just track the last epoch for reference
            if not hasattr(self, '_last_epoch'):
                self._last_epoch = -1
            
            self._last_epoch = epoch
            
            # Get basic metrics from logs
            train_loss = logs.get('loss', 0.0)
            train_accuracy = logs.get('accuracy', 0.0)
            val_loss = logs.get('val_loss', 0.0)
            val_accuracy = logs.get('val_accuracy', 0.0)
            
            # Try to compute additional metrics from validation data
            # We'll load validation data on-the-fly
            try:
                metrics = self._compute_metrics_from_path(epoch)
            except Exception as e:
                logger.debug(f"Could not compute additional metrics: {e}")
                metrics = {}
            
            # Extract learning rate from optimizer if available
            learning_rate = None
            if hasattr(self, 'model') and self.model and hasattr(self.model, 'optimizer'):
                try:
                    lr_var = self.model.optimizer.learning_rate
                    if hasattr(lr_var, 'numpy'):
                        learning_rate = float(lr_var.numpy())
                    elif hasattr(lr_var, 'value'):
                        learning_rate = float(lr_var.value())
                    else:
                        learning_rate = float(lr_var)
                except:
                    pass
            
            # Create epoch metric
            epoch_metric = {
                'trial': self.current_trial if self.current_trial is not None else 1,
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'learning_rate': learning_rate,
                **metrics
            }
            
            self.epoch_metrics.append(epoch_metric)
            
            # Save to file incrementally
            self._save_metrics()
            
        except Exception as e:
            logger.warning(f"Error in LightweightMetricsTracker.on_epoch_end: {e}")
    
    def _compute_metrics_from_path(self, epoch: int) -> Dict:
        """
        Compute metrics by loading validation data from path.
        This is slower but allows the callback to be deep-copyable.
        """
        if not self.model or not hasattr(self, 'model'):
            return {}
        
        try:
            # Load validation data using ImageDataGenerator
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
            
            val_datagen = ImageDataGenerator(
                rescale=1./255,
                validation_split=self.validation_split if self.use_validation_subset else 0.0
            )
            
            if self.use_validation_subset:
                # Validation is a subset of training data
                val_gen = val_datagen.flow_from_directory(
                    self.val_data_path,
                    target_size=(224, 224),  # Default, could be made configurable
                    batch_size=32,
                    class_mode='categorical',
                    subset='validation',
                    shuffle=False,
                    seed=42
                )
            else:
                # Validation is in a separate directory
                val_gen = val_datagen.flow_from_directory(
                    self.val_data_path,
                    target_size=(224, 224),  # Default, could be made configurable
                    batch_size=32,
                    class_mode='categorical',
                    shuffle=False,
                    seed=42
                )
            
            # Collect predictions and true labels
            y_true = []
            y_pred_proba = []
            
            for i in range(len(val_gen)):
                batch_x, batch_y = val_gen[i]
                batch_pred = self.model.predict(batch_x, verbose=0)
                y_true.append(batch_y)
                y_pred_proba.append(batch_pred)
            
            # Concatenate
            y_true = np.concatenate(y_true, axis=0)
            y_pred_proba = np.concatenate(y_pred_proba, axis=0)
            y_pred = np.argmax(y_pred_proba, axis=1)
            y_true_classes = np.argmax(y_true, axis=1)
            
            # Compute metrics
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UndefinedMetricWarning)
                accuracy = accuracy_score(y_true_classes, y_pred)
                precision = precision_score(y_true_classes, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_true_classes, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_true_classes, y_pred, average='weighted', zero_division=0)
                
                # AUC (handle binary and multiclass)
                if self.num_classes == 2:
                    auc = roc_auc_score(y_true_classes, y_pred_proba[:, 1])
                else:
                    auc = roc_auc_score(y_true_classes, y_pred_proba, multi_class='ovr', average='weighted')
                
                # Per-class metrics
                report = classification_report(y_true_classes, y_pred, 
                                               target_names=self.class_names,
                                               output_dict=True, zero_division=0)
                
                per_class_precision = [report.get(name, {}).get('precision', 0.0) 
                                      for name in self.class_names]
                per_class_recall = [report.get(name, {}).get('recall', 0.0) 
                                   for name in self.class_names]
                per_class_f1 = [report.get(name, {}).get('f1-score', 0.0) 
                               for name in self.class_names]
                per_class_support = [report.get(name, {}).get('support', 0) 
                                    for name in self.class_names]
            
            return {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'accuracy': accuracy,
                'auc': auc,
                'per_class_precision': per_class_precision,
                'per_class_recall': per_class_recall,
                'per_class_f1': per_class_f1,
                'per_class_support': per_class_support
            }
        except Exception as e:
            logger.debug(f"Could not compute metrics from path: {e}")
            return {}
    
    def _save_metrics(self):
        """Save metrics to JSON file, appending to existing metrics from previous trials."""
        try:
            # Load existing metrics if file exists
            existing_metrics = []
            if self.output_file.exists():
                try:
                    with open(self.output_file, 'r') as f:
                        existing_data = json.load(f)
                        existing_metrics = existing_data.get('epoch_metrics', [])
                except Exception as e:
                    logger.debug(f"Could not load existing metrics: {e}")
                    existing_metrics = []
            
            # Merge existing metrics with current metrics (avoid duplicates)
            # Use trial and epoch as unique identifiers
            existing_keys = {(m.get('trial', 1), m.get('epoch', 0)) for m in existing_metrics}
            new_metrics = [m for m in self.epoch_metrics 
                          if (m.get('trial', 1), m.get('epoch', 0)) not in existing_keys]
            
            # Combine all metrics
            all_metrics = existing_metrics + new_metrics
            
            # Save all metrics
            metrics_data = {
                'epoch_metrics': all_metrics
            }
            with open(self.output_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save metrics: {e}")

