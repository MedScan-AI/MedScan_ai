"""
Training script for Custom CNN model.
Trains Custom CNN for Tuberculosis (TB) and Lung Cancer detection.
Uses TensorFlow for model training.
"""

import os
import sys
import json
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import argparse

import numpy as np
import pandas as pd
import yaml
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
# Custom CNN doesn't use pre-trained models from applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
import shutil
import tempfile
import mlflow
import mlflow.tensorflow
import keras_tuner as kt

# Suppress TensorFlow retracing warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING
warnings.filterwarnings('ignore', message='.*retracing.*')
warnings.filterwarnings('ignore', message='.*tf.function.*')
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

# Suppress Keras PyDataset warning
warnings.filterwarnings('ignore', message='.*PyDataset.*super.*__init__.*')
warnings.filterwarnings('ignore', category=UserWarning, module='keras.*py_dataset_adapter')

# Configure logging
# Set TensorFlow logger to ERROR level to suppress retracing warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import bias detection module (after logger is configured)
try:
    from bias_detection import BiasDetector
    BIAS_DETECTION_AVAILABLE = True
except ImportError:
    BIAS_DETECTION_AVAILABLE = False
    logger.warning("Bias detection module not available. Skipping bias analysis.")

# Import interpretability module (after logger is configured)
try:
    from interpretability import ModelInterpreter
    INTERPRETABILITY_AVAILABLE = True
except ImportError:
    INTERPRETABILITY_AVAILABLE = False
    logger.warning("Interpretability module not available. Skipping SHAP/LIME analysis.")

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)


class ConfigLoader:
    """Load and manage configuration from YAML file."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize ConfigLoader.
        
        Args:
            config_path: Path to config YAML file. If None, uses default path.
        """
        if config_path is None:
            # Default to ModelDevelopment/config/vision_training.yml
            script_dir = Path(__file__).parent
            script_abs = script_dir.absolute()
            
            # Detect if running in Docker (script is in /app)
            is_docker = str(script_abs).startswith('/app')
            
            # Try multiple locations in order:
            if is_docker:
                # In Docker: check /app/config first, then relative paths
                possible_paths = [
                    Path("/app/config/vision_training.yml"),  # Docker absolute path (most reliable)
                    script_abs / "config" / "vision_training.yml",  # /app/config (same as above but relative)
                ]
            else:
                # Local development: check parent directory config
                possible_paths = [
                    script_abs.parent / "config" / "vision_training.yml",  # ModelDevelopment/config
                    Path("/app/config/vision_training.yml"),  # Also check Docker path (in case)
                ]
            
            # Remove duplicates while preserving order
            seen = set()
            unique_paths = []
            for path in possible_paths:
                path_str = str(path)
                if path_str not in seen:
                    seen.add(path_str)
                    unique_paths.append(path)
            
            for path in unique_paths:
                if path.exists():
                    config_path = path
                    break
            else:
                # If none found, use appropriate default based on environment
                if is_docker:
                    config_path = Path("/app/config/vision_training.yml")
                else:
                    config_path = script_abs.parent / "config" / "vision_training.yml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        logger.info(f"Loaded configuration from {self.config_path}")
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            # Provide helpful error message with suggestions
            error_msg = f"Config file not found: {self.config_path}\n"
            error_msg += "Please ensure:\n"
            error_msg += "  1. The config file exists at the specified path\n"
            error_msg += "  2. If running in Docker, the config directory is mounted as a volume\n"
            error_msg += "  3. Use --config flag to specify the config file path explicitly\n"
            error_msg += f"   Expected locations:\n"
            error_msg += f"   - /app/config/vision_training.yml (Docker)\n"
            error_msg += f"   - {Path(__file__).parent.parent / 'config' / 'vision_training.yml'} (Local)"
            raise FileNotFoundError(error_msg)
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get(self, key_path: str, default=None):
        """
        Get configuration value using dot notation (e.g., 'training.epochs').
        
        Args:
            key_path: Dot-separated path to config value
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_path(self, key: str) -> Path:
        """Get path configuration and return as Path object."""
        path_str = self.get(f"paths.{key}")
        if path_str:
            return Path(path_str)
        return None


class DataLoader:
    """Load preprocessed image data."""
    
    def __init__(self, base_path: str, dataset_name: str, config: Optional[ConfigLoader] = None):
        """
        Initialize DataLoader.
        
        Args:
            base_path: Base path to preprocessed data
            dataset_name: Name of dataset ('tb' or 'lung_cancer_ct_scan')
            config: Optional ConfigLoader instance
        """
        self.base_path = Path(base_path)
        self.dataset_name = dataset_name
        self.config = config
        self.data_path = self._find_latest_partition()
        logger.info(f"Using data path: {self.data_path}")
        
    def _find_latest_partition(self) -> Path:
        """Find the latest partition directory (YYYY/MM/DD)."""
        dataset_path = self.base_path / self.dataset_name
        if not dataset_path.exists():
            raise ValueError(f"Dataset path not found: {dataset_path}")
        
        # Find latest year
        years = [d for d in dataset_path.iterdir() if d.is_dir() and d.name.isdigit()]
        if not years:
            raise ValueError(f"No year directories found in {dataset_path}")
        latest_year = max(years, key=lambda x: int(x.name))
        
        # Find latest month
        months = [d for d in latest_year.iterdir() if d.is_dir() and d.name.isdigit()]
        if not months:
            raise ValueError(f"No month directories found in {latest_year}")
        latest_month = max(months, key=lambda x: int(x.name))
        
        # Find latest day
        days = [d for d in latest_month.iterdir() if d.is_dir() and d.name.isdigit()]
        if not days:
            raise ValueError(f"No day directories found in {latest_month}")
        latest_day = max(days, key=lambda x: int(x.name))
        
        return latest_day
    
    def get_classes(self) -> List[str]:
        """Get class names from train directory."""
        train_path = self.data_path / "train"
        if not train_path.exists():
            raise ValueError(f"Train directory not found: {train_path}")
        
        classes = [d.name for d in train_path.iterdir() if d.is_dir()]
        classes.sort()
        logger.info(f"Found classes: {classes}")
        return classes
    
    def _sample_files_from_directory(
        self,
        directory: Path,
        percent_use: float,
        seed: int
    ) -> Path:
        """
        Sample a subset of files from a directory structure (with class subdirectories).
        Creates a temporary directory with sampled files maintaining the same structure.
        
        Args:
            directory: Source directory with class subdirectories
            percent_use: Percentage of files to use (0.0 to 1.0)
            seed: Random seed for reproducibility
            
        Returns:
            Path to temporary directory with sampled files, or original directory if percent_use == 1.0
        """
        if percent_use >= 1.0:
            return directory
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Create temporary directory
        temp_dir = Path(tempfile.mkdtemp(prefix="data_subset_"))
        logger.info(f"Creating data subset ({percent_use*100:.1f}%) in temporary directory: {temp_dir}")
        
        # Get all class directories
        class_dirs = [d for d in directory.iterdir() if d.is_dir()]
        
        total_files = 0
        sampled_files = 0
        
        for class_dir in class_dirs:
            # Get all image files in this class directory
            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + \
                         list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.JPG")) + \
                         list(class_dir.glob("*.PNG"))
            
            if len(image_files) == 0:
                continue
            
            total_files += len(image_files)
            
            # Sample files
            num_samples = max(1, int(len(image_files) * percent_use))
            sampled = random.sample(image_files, num_samples)
            sampled_files += len(sampled)
            
            # Create class directory in temp location
            temp_class_dir = temp_dir / class_dir.name
            temp_class_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy sampled files (using symlinks for efficiency, fallback to copy)
            for src_file in sampled:
                dst_file = temp_class_dir / src_file.name
                try:
                    dst_file.symlink_to(src_file)
                except (OSError, NotImplementedError):
                    # Fallback to copy if symlinks not supported (Windows)
                    shutil.copy2(src_file, dst_file)
        
        logger.info(f"Sampled {sampled_files}/{total_files} files ({sampled_files/total_files*100:.1f}%)")
        
        # Store temp directory for cleanup later
        self._temp_dirs = getattr(self, '_temp_dirs', [])
        self._temp_dirs.append(temp_dir)
        
        return temp_dir
    
    def create_data_generators(
        self,
        batch_size: int = 32,
        image_size: Tuple[int, int] = (224, 224),
        validation_split: float = None,
        max_samples: Optional[int] = None
    ) -> Tuple[ImageDataGenerator, ImageDataGenerator, ImageDataGenerator]:
        """
        Create data generators for train, validation, and test sets.
        
        Args:
            batch_size: Batch size for training
            image_size: Target image size (height, width)
            validation_split: Fraction of training data to use for validation
            max_samples: Maximum number of samples per split (for dry run)
            
        Returns:
            Tuple of (train_gen, val_gen, test_gen)
        """
        # Get augmentation config
        if self.config:
            aug_config = self.config.get('data_augmentation', {})
            if validation_split is None:
                validation_split = self.config.get('training.validation_split', 0.2)
        else:
            aug_config = {}
            if validation_split is None:
                validation_split = 0.2
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1.0/255.0,
            rotation_range=aug_config.get('rotation_range', 20),
            width_shift_range=aug_config.get('width_shift_range', 0.2),
            height_shift_range=aug_config.get('height_shift_range', 0.2),
            horizontal_flip=aug_config.get('horizontal_flip', True),
            zoom_range=aug_config.get('zoom_range', 0.2),
            fill_mode=aug_config.get('fill_mode', 'nearest'),
            validation_split=validation_split
        )
        
        # No augmentation for validation and test
        val_test_datagen = ImageDataGenerator(
            rescale=1.0/255.0,
            validation_split=validation_split
        )
        
        # Get data subset configuration
        data_percent_use = 1.0
        data_seed = 42
        if self.config:
            data_percent_use = self.config.get('training.data_percent_use', 1.0)
            data_seed = self.config.get('training.data_seed', 42)
        
        # Sample data if needed
        original_train_path = self.data_path / "train"
        train_path = original_train_path
        test_path = self.data_path / "test"
        val_path = self.data_path / "valid" if (self.data_path / "valid").exists() else None
        
        if data_percent_use < 1.0:
            logger.info(f"Using {data_percent_use*100:.1f}% of data (seed={data_seed})")
            train_path = self._sample_files_from_directory(train_path, data_percent_use, data_seed)
            test_path = self._sample_files_from_directory(test_path, data_percent_use, data_seed)
            if val_path:
                val_path = self._sample_files_from_directory(val_path, data_percent_use, data_seed)
        
        # Get classes from the (possibly sampled) train path
        classes = [d.name for d in train_path.iterdir() if d.is_dir()]
        classes.sort()
        logger.info(f"Found classes: {classes}")
        num_classes = len(classes)
        
        # Training generator
        train_gen = train_datagen.flow_from_directory(
            train_path,
            target_size=image_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True,
            seed=42
        )
        
        # Validation generator (from train split if no separate val folder)
        if val_path and val_path.exists():
            val_gen = val_test_datagen.flow_from_directory(
                val_path,
                target_size=image_size,
                batch_size=batch_size,
                class_mode='categorical',
                shuffle=False,
                seed=42
            )
        else:
            val_gen = train_datagen.flow_from_directory(
                train_path,
                target_size=image_size,
                batch_size=batch_size,
                class_mode='categorical',
                subset='validation',
                shuffle=False,
                seed=42
            )
        
        # Test generator
        test_gen = val_test_datagen.flow_from_directory(
            test_path,
            target_size=image_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False,
            seed=42
        )
        
        # Limit samples for dry run if specified
        # Store the limited sample counts as attributes for use in training
        if max_samples is not None:
            # Calculate distribution: 50% train, 25% val, 25% test
            train_limit = max_samples // 2
            val_limit = max_samples // 4
            test_limit = max_samples // 4
            
            # Store limited sample counts as attributes on the generators
            train_gen._limited_samples = min(train_gen.samples, train_limit)
            val_gen._limited_samples = min(val_gen.samples, val_limit)
            test_gen._limited_samples = min(test_gen.samples, test_limit)
            
            # Calculate steps per epoch for limited samples
            train_gen._steps_per_epoch = (train_gen._limited_samples + train_gen.batch_size - 1) // train_gen.batch_size
            val_gen._steps_per_epoch = (val_gen._limited_samples + val_gen.batch_size - 1) // val_gen.batch_size
            test_gen._steps_per_epoch = (test_gen._limited_samples + test_gen.batch_size - 1) // test_gen.batch_size
            
            logger.info(f"DRY RUN MODE: Limited samples (train: {train_gen._limited_samples}, val: {val_gen._limited_samples}, test: {test_gen._limited_samples})")
        
        logger.info(f"Train samples: {train_gen.samples}")
        logger.info(f"Validation samples: {val_gen.samples}")
        logger.info(f"Test samples: {test_gen.samples}")
        
        return train_gen, val_gen, test_gen


class ModelBuilder:
    """Build Custom CNN model architecture."""
    
    def __init__(self, config: ConfigLoader):
        """Initialize ModelBuilder with configuration."""
        self.config = config
    
    def build_cnn_custom(
        self,
        input_shape: Tuple[int, int, int],
        num_classes: int
    ) -> models.Model:
        """
        Build custom CNN model.
        
        Args:
            input_shape: Input image shape (height, width, channels)
            num_classes: Number of output classes
            
        Returns:
            Compiled Keras model
        """
        cnn_config = self.config.get('models.cnn_custom', {})
        conv_blocks = cnn_config.get('conv_blocks', [])
        dense_layers = cnn_config.get('dense_layers', [])
        
        # Use Input layer as first layer (recommended approach to avoid warning)
        inputs = keras.Input(shape=input_shape)
        x = inputs
        
        # Add convolutional blocks
        for i, block in enumerate(conv_blocks):
            filters = block.get('filters', 32 * (2 ** i))
            kernel_size = tuple(block.get('kernel_size', [3, 3]))
            dropout = block.get('dropout', 0.25)
            
            x = layers.Conv2D(filters, kernel_size, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Dropout(dropout)(x)
        
        # Flatten
        x = layers.Flatten()(x)
        
        # Add dense layers
        for layer in dense_layers:
            units = layer.get('units', 512)
            dropout = layer.get('dropout', 0.5)
            x = layers.Dense(units, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(dropout)(x)
        
        # Output layer
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        # Create model using functional API
        model = models.Model(inputs=inputs, outputs=outputs, name='custom_cnn')
        
        learning_rate = cnn_config.get('learning_rate', 0.001)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        return model


class ModelTrainer:
    """Train and evaluate models."""
    
    def __init__(
        self,
        model: models.Model,
        model_name: str,
        models_dir: Path,
        logs_dir: Path,
        dataset_name: str,
        config: Optional[ConfigLoader] = None
    ):
        """
        Initialize ModelTrainer.
        
        Args:
            model: Keras model to train
            model_name: Name of the model architecture
            models_dir: Directory to save models
            logs_dir: Directory to save training logs
            dataset_name: Name of the dataset
            config: Optional ConfigLoader instance
        """
        self.model = model
        self.model_name = model_name
        self.models_dir = models_dir
        self.logs_dir = logs_dir
        self.dataset_name = dataset_name
        self.config = config
        self.history = None
        
    def train(
        self,
        train_gen: ImageDataGenerator,
        val_gen: ImageDataGenerator,
        epochs: int = 50,
        early_stopping_patience: int = None,
        reduce_lr_patience: int = None
    ) -> Dict:
        """
        Train the model.
        
        Args:
            train_gen: Training data generator
            val_gen: Validation data generator
            epochs: Number of training epochs
            early_stopping_patience: Patience for early stopping (from config if None)
            reduce_lr_patience: Patience for learning rate reduction (from config if None)
            
        Returns:
            Training history dictionary
        """
        # Get callback configs
        checkpoint_config = self.config.get('callbacks.model_checkpoint', {}) if self.config else {}
        early_stop_config = self.config.get('callbacks.early_stopping', {}) if self.config else {}
        reduce_lr_config = self.config.get('callbacks.reduce_lr', {}) if self.config else {}
        
        if early_stopping_patience is None:
            early_stopping_patience = early_stop_config.get('patience', 10)
        if reduce_lr_patience is None:
            reduce_lr_patience = reduce_lr_config.get('patience', 5)
        
        # Create callbacks
        model_checkpoint = callbacks.ModelCheckpoint(
            filepath=str(self.models_dir / f"{self.model_name}_best.keras"),
            monitor=checkpoint_config.get('monitor', 'val_accuracy'),
            save_best_only=checkpoint_config.get('save_best_only', True),
            mode=checkpoint_config.get('mode', 'max'),
            verbose=checkpoint_config.get('verbose', 1)
        )
        
        early_stopping = callbacks.EarlyStopping(
            monitor=early_stop_config.get('monitor', 'val_accuracy'),
            patience=early_stopping_patience,
            min_delta=early_stop_config.get('min_delta', 0.0),
            restore_best_weights=early_stop_config.get('restore_best_weights', True),
            verbose=early_stop_config.get('verbose', 1)
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor=reduce_lr_config.get('monitor', 'val_loss'),
            factor=reduce_lr_config.get('factor', 0.5),
            patience=reduce_lr_patience,
            min_lr=reduce_lr_config.get('min_lr', 1e-7),
            verbose=reduce_lr_config.get('verbose', 1)
        )
        
        csv_logger = callbacks.CSVLogger(
            str(self.logs_dir / f"{self.model_name}_training.log")
        )
        
        # Train model
        logger.info(f"Training {self.model_name}...")
        
        # Calculate steps per epoch if generator has limited samples (for dry run)
        steps_per_epoch = None
        validation_steps = None
        if hasattr(train_gen, '_steps_per_epoch'):
            # Use limited steps for dry run
            steps_per_epoch = train_gen._steps_per_epoch
        elif hasattr(train_gen, 'samples') and train_gen.samples > 0:
            steps_per_epoch = (train_gen.samples + train_gen.batch_size - 1) // train_gen.batch_size
        
        if hasattr(val_gen, '_steps_per_epoch'):
            # Use limited steps for dry run
            validation_steps = val_gen._steps_per_epoch
        elif hasattr(val_gen, 'samples') and val_gen.samples > 0:
            validation_steps = (val_gen.samples + val_gen.batch_size - 1) // val_gen.batch_size
        
        fit_kwargs = {
            'epochs': epochs,
            'validation_data': val_gen,
            'callbacks': [model_checkpoint, early_stopping, reduce_lr, csv_logger],
            'verbose': 1
        }
        
        if steps_per_epoch is not None:
            fit_kwargs['steps_per_epoch'] = steps_per_epoch
        if validation_steps is not None:
            fit_kwargs['validation_steps'] = validation_steps
        
        self.history = self.model.fit(train_gen, **fit_kwargs)
        
        return self.history.history
    
    def evaluate(
        self,
        test_gen: ImageDataGenerator
    ) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            test_gen: Test data generator
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating {self.model_name} on test set...")
        
        # Calculate steps if generator has limited samples (for dry run)
        steps = None
        if hasattr(test_gen, '_steps_per_epoch'):
            # Use limited steps for dry run
            steps = test_gen._steps_per_epoch
        elif hasattr(test_gen, 'samples') and test_gen.samples > 0:
            steps = (test_gen.samples + test_gen.batch_size - 1) // test_gen.batch_size
        
        evaluate_kwargs = {'verbose': 1}
        if steps is not None:
            evaluate_kwargs['steps'] = steps
        
        results = self.model.evaluate(test_gen, **evaluate_kwargs)
        
        metrics = {
            'test_loss': results[0],
            'test_accuracy': results[1],
            'test_top_k_accuracy': results[2] if len(results) > 2 else None
        }
        
        return metrics
    
    def save_model(self, path: Optional[Path] = None):
        """Save the trained model."""
        if path is None:
            path = self.models_dir / f"{self.model_name}_final.keras"
        
        self.model.save(str(path))
        logger.info(f"Model saved to {path}")


class HyperparameterTuner:
    """Hyperparameter tuning using KerasTuner."""
    
    def __init__(self, config: ConfigLoader, model_builder: 'ModelBuilder', dataset_name: str = None):
        """
        Initialize HyperparameterTuner.
        
        Args:
            config: ConfigLoader instance
            model_builder: ModelBuilder instance
            dataset_name: Name of the dataset (for project name isolation)
        """
        self.config = config
        self.model_builder = model_builder
        self.tuning_config = config.get('hyperparameter_tuning', {})
        self.dataset_name = dataset_name or 'default'
    
    def _build_tunable_cnn_custom(self, hp, input_shape: Tuple[int, int, int], num_classes: int) -> models.Model:
        """Build tunable Custom CNN model."""
        search_space = self.tuning_config.get('search_space', {})
        cnn_space = search_space.get('cnn_custom', {})
        common_space = search_space
        
        learning_rate = hp.Float(
            'learning_rate',
            min_value=common_space.get('learning_rate', {}).get('min', 0.0001),
            max_value=common_space.get('learning_rate', {}).get('max', 0.01),
            step=common_space.get('learning_rate', {}).get('step', 0.0001)
        )
        
        multiplier = hp.Choice(
            'conv_filters_multiplier',
            values=cnn_space.get('conv_filters_multiplier', {}).get('values', [1.0, 1.5, 2.0])
        )
        
        dense_units_1 = hp.Choice(
            'dense_units_1',
            values=cnn_space.get('dense_units_1', {}).get('values', [256, 512, 1024])
        )
        
        dense_units_2 = hp.Choice(
            'dense_units_2',
            values=cnn_space.get('dense_units_2', {}).get('values', [128, 256, 512])
        )
        
        dropout = hp.Float(
            'dropout',
            min_value=common_space.get('dropout_rate', {}).get('min', 0.2),
            max_value=common_space.get('dropout_rate', {}).get('max', 0.6),
            step=common_space.get('dropout_rate', {}).get('step', 0.1)
        )
        
        # Build model with tunable parameters using functional API
        # Use Input layer as first layer (recommended approach to avoid warning)
        inputs = keras.Input(shape=input_shape)
        x = inputs
        
        base_filters = [32, 64, 128, 256]
        
        for i, base_filter in enumerate(base_filters):
            filters = int(base_filter * multiplier)
            x = layers.Conv2D(filters, (3, 3), activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Dropout(dropout)(x)
        
        x = layers.Flatten()(x)
        x = layers.Dense(dense_units_1, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Dense(dense_units_2, activation='relu')(x)
        x = layers.Dropout(dropout * 0.6)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        # Create model using functional API
        model = models.Model(inputs=inputs, outputs=outputs, name='tunable_custom_cnn')
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        return model
    
    def tune_model(
        self,
        model_name: str,
        input_shape: Tuple[int, int, int],
        num_classes: int,
        train_gen: ImageDataGenerator,
        val_gen: ImageDataGenerator,
        epochs: int,
        output_dir: Path
    ) -> models.Model:
        """
        Tune hyperparameters for Custom CNN model.
        
        Args:
            model_name: Name of the model to tune
            input_shape: Input image shape
            num_classes: Number of classes
            train_gen: Training data generator
            val_gen: Validation data generator
            epochs: Number of epochs per trial
            output_dir: Directory to save tuning results
            
        Returns:
            Best model with tuned hyperparameters
        """
        tuner_type = self.tuning_config.get('tuner_type', 'random')
        max_trials = self.tuning_config.get('max_trials', 10)
        executions_per_trial = self.tuning_config.get('executions_per_trial', 1)
        objective = self.tuning_config.get('objective', 'val_accuracy')
        direction = self.tuning_config.get('direction', 'max')
        directory = Path(self.tuning_config.get('directory', '../data/hyperparameter_tuning'))
        # Include dataset name in project name to avoid conflicts between datasets
        dataset_name = getattr(self, 'dataset_name', 'default')
        project_name = f"{self.tuning_config.get('project_name', 'vision_model_tuning')}_{model_name}_{dataset_name}"
        
        # Create tuner
        model_builder_fn = lambda hp: self._build_tunable_cnn_custom(hp, input_shape, num_classes)
        
        if tuner_type == 'random':
            tuner = kt.RandomSearch(
                model_builder_fn,
                objective=kt.Objective(objective, direction),
                max_trials=max_trials,
                executions_per_trial=executions_per_trial,
                directory=str(directory),
                project_name=project_name
            )
        elif tuner_type == 'bayesian':
            tuner = kt.BayesianOptimization(
                model_builder_fn,
                objective=kt.Objective(objective, direction),
                max_trials=max_trials,
                executions_per_trial=executions_per_trial,
                directory=str(directory),
                project_name=project_name
            )
        elif tuner_type == 'hyperband':
            tuner = kt.Hyperband(
                model_builder_fn,
                objective=kt.Objective(objective, direction),
                max_epochs=epochs,
                directory=str(directory),
                project_name=project_name
            )
        else:
            raise ValueError(f"Unknown tuner type: {tuner_type}")
        
        # Create callbacks for hyperparameter tuning (early stopping to save time)
        early_stop_config = self.config.get('callbacks.early_stopping', {})
        reduce_lr_config = self.config.get('callbacks.reduce_lr', {})
        
        tuning_callbacks = [
            callbacks.EarlyStopping(
                monitor=early_stop_config.get('monitor', 'val_accuracy'),
                patience=early_stop_config.get('patience', 10),
                min_delta=early_stop_config.get('min_delta', 0.0),
                restore_best_weights=early_stop_config.get('restore_best_weights', True),
                verbose=early_stop_config.get('verbose', 1)
            ),
            callbacks.ReduceLROnPlateau(
                monitor=reduce_lr_config.get('monitor', 'val_loss'),
                factor=reduce_lr_config.get('factor', 0.5),
                patience=reduce_lr_config.get('patience', 5),
                min_lr=reduce_lr_config.get('min_lr', 1e-7),
                verbose=reduce_lr_config.get('verbose', 1)
            )
        ]
        
        # Run hyperparameter search
        logger.info(f"Starting hyperparameter tuning for {model_name}...")
        logger.info(f"Tuner type: {tuner_type}, Max trials: {max_trials}")
        logger.info(f"Early stopping enabled during tuning (patience: {early_stop_config.get('patience', 10)})")
        
        tuner.search(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=tuning_callbacks,
            verbose=1
        )
        
        # Get best model
        best_model = tuner.get_best_models(num_models=1)[0]
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        
        logger.info(f"Best hyperparameters for {model_name}:")
        for param, value in best_hps.values.items():
            logger.info(f"  {param}: {value}")
        
        # Print model parameters after tuning
        total_params = best_model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in best_model.trainable_weights])
        non_trainable_params = total_params - trainable_params
        logger.info(f"\n{'='*60}")
        logger.info(f"Model Parameters Summary for {model_name} (After Tuning)")
        logger.info(f"{'='*60}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Non-trainable parameters: {non_trainable_params:,}")
        logger.info(f"{'='*60}\n")
        
        # Save best hyperparameters
        best_hps_file = output_dir / f"{model_name}_best_hyperparameters.json"
        with open(best_hps_file, 'w') as f:
            json.dump(best_hps.values, f, indent=2)
        
        return best_model


def train_dataset(
    dataset_name: str,
    base_data_path: str,
    output_base_path: str,
    config: ConfigLoader,
    epochs: int = None,
    batch_size: int = None,
    image_size: Tuple[int, int] = None,
    dry_run: bool = False
):
    """
    Train Custom CNN model for a specific dataset.
    
    Args:
        dataset_name: Name of dataset ('tb' or 'lung_cancer_ct_scan')
        base_data_path: Base path to preprocessed data
        output_base_path: Base path for model outputs (should be ModelDevelopment/data)
        config: ConfigLoader instance
        epochs: Number of training epochs (from config if None)
        batch_size: Batch size for training (from config if None)
        image_size: Target image size (from config if None)
        dry_run: If True, limit to 64 images for quick testing
    """
    if dry_run:
        logger.info("="*60)
        logger.info("DRY RUN MODE: Using only 64 images for quick testing")
        logger.info("="*60)
    
    logger.info(f"Starting Custom CNN training for {dataset_name} dataset")
    
    # Get training config
    training_config = config.get('training', {})
    if epochs is None:
        epochs = training_config.get('epochs', 50)
    if batch_size is None:
        batch_size = training_config.get('batch_size', 32)
    if image_size is None:
        image_size = tuple(training_config.get('image_size', [224, 224]))
    
    # Setup directory structure with date partitions (year/month/day/timestamp)
    now = datetime.now()
    year = now.strftime("%Y")
    month = now.strftime("%m")
    day = now.strftime("%d")
    timestamp = now.strftime("%H%M%S")
    
    # Set model name early for directory structure
    model_name = 'CNN_Custom'
    
    data_dir = Path(output_base_path)
    # Add dataset and model name as additional partition
    models_dir = data_dir / "models" / year / month / day / timestamp / f"{dataset_name}_{model_name}"
    logs_dir = data_dir / "logs" / year / month / day / timestamp
    metadata_dir = data_dir / year / month / day / timestamp
    mlflow_dir = data_dir / "mlruns"
    
    # Create directories
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    mlflow_dir.mkdir(parents=True, exist_ok=True)
    
    # Set MLflow tracking URI
    mlflow_tracking_uri = f"file:///{mlflow_dir.absolute()}"
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    logger.info(f"MLflow tracking URI: {mlflow_tracking_uri}")
    
    # Load data
    data_loader = DataLoader(base_data_path, dataset_name, config)
    # data_loader.data_path is already the latest day directory
    latest_day = data_loader.data_path
    max_samples = 64 if dry_run else None
    train_gen, val_gen, test_gen = data_loader.create_data_generators(
        batch_size=batch_size,
        image_size=image_size,
        max_samples=max_samples
    )
    
    # Get number of classes
    num_classes = len(data_loader.get_classes())
    input_shape = (*image_size, 3)
    
    # Initialize MLflow
    mlflow.set_experiment(f"{dataset_name}_custom_cnn_training")
    
    # Check if hyperparameter tuning is enabled
    tuning_enabled = config.get('hyperparameter_tuning.enabled', False)
    
    # Build model
    model_builder = ModelBuilder(config)
    hyperparameter_tuner = HyperparameterTuner(config, model_builder, dataset_name) if tuning_enabled else None
    
    if tuning_enabled:
        logger.info("Hyperparameter tuning is enabled. Tuning Custom CNN...")
        model = hyperparameter_tuner.tune_model(
            model_name=model_name,
            input_shape=input_shape,
            num_classes=num_classes,
            train_gen=train_gen,
            val_gen=val_gen,
            epochs=epochs,
            output_dir=metadata_dir
        )
        tuned = True
    else:
        model = model_builder.build_cnn_custom(input_shape, num_classes)
        tuned = False
    
    # Print model parameters before training (for both tuned and non-tuned models)
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    logger.info(f"\n{'='*60}")
    logger.info(f"Model Parameters Summary for {model_name}")
    logger.info(f"{'='*60}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Non-trainable parameters: {non_trainable_params:,}")
    logger.info(f"{'='*60}\n")
    
    # Train and evaluate model
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {model_name}")
    logger.info(f"{'='*60}\n")
    
    with mlflow.start_run(run_name=f"{dataset_name}_{model_name}"):
        # Log parameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("dataset", dataset_name)
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("image_size", image_size)
        mlflow.log_param("hyperparameter_tuning", tuning_enabled)
        
        # Create trainer
        trainer = ModelTrainer(
            model=model,
            model_name=model_name,
            models_dir=models_dir,
            logs_dir=logs_dir,
            dataset_name=dataset_name,
            config=config
        )
        
        # Train model (skip if already trained during hyperparameter tuning)
        if tuned:
            logger.info(f"{model_name} was already trained during hyperparameter tuning. Skipping additional training.")
            history = {
                'loss': [0.0],
                'accuracy': [0.0],
                'val_loss': [0.0],
                'val_accuracy': [0.0]
            }
        else:
            # Train model
            # For dry run, limit epochs to 2 for quick testing
            training_epochs = 2 if dry_run else epochs
            if dry_run:
                logger.info(f"DRY RUN: Limiting training to {training_epochs} epochs")
            history = trainer.train(train_gen, val_gen, epochs=training_epochs)
        
        # Evaluate model
        metrics = trainer.evaluate(test_gen)
        
        # Log metrics
        for key, value in metrics.items():
            if value is not None:
                mlflow.log_metric(key, value)
        
        # Bias Detection
        bias_config = config.get('bias_detection', {})
        if bias_config.get('enabled', False) and BIAS_DETECTION_AVAILABLE:
            logger.info("\n" + "="*60)
            logger.info("Running Bias Detection Analysis...")
            logger.info("="*60)
            
            try:
                # Get data path for bias detection
                # For sampled data, we need to pass both:
                # 1. The sampled test directory (for loading images)
                # 2. The original data path (for loading metadata)
                if hasattr(test_gen, '_sampled_path') and test_gen._sampled_path:
                    # The sampled path is the test directory itself
                    sampled_test_path = test_gen._sampled_path
                    # Original data path for metadata lookup (latest_day is already the full path)
                    original_data_path = latest_day
                    logger.info(f"Using sampled test directory for images: {sampled_test_path}")
                    logger.info(f"Using original data path for metadata: {original_data_path}")
                else:
                    # Use original data path for both images and metadata (latest_day is already the full path)
                    sampled_test_path = None
                    original_data_path = latest_day
                    logger.info(f"Using original data path for bias detection: {original_data_path}")
                
                # Create bias detector
                # Pass sampled test path if available, otherwise use original data path
                bias_detector = BiasDetector(
                    model=trainer.model,
                    data_path=sampled_test_path if sampled_test_path else original_data_path,
                    dataset_name=dataset_name,
                    output_dir=metadata_dir / "bias_reports",
                    config=config.config,
                    metadata_path=original_data_path  # Always use original path for metadata
                )
                
                # Run bias detection (and mitigation if enabled) on test set
                max_samples = 64 if dry_run else None
                mitigation_enabled = bias_config.get('mitigation_enabled', False)
                
                logger.info(f"Bias detection enabled: {bias_config.get('enabled', False)}")
                logger.info(f"Bias mitigation enabled: {mitigation_enabled}")
                
                if mitigation_enabled:
                    logger.info("Running bias detection with mitigation enabled...")
                    bias_results = bias_detector.detect_and_mitigate_bias(
                        split='test',
                        max_samples=max_samples,
                        apply_mitigation=True
                    )
                    # Extract original results for logging (mitigation results are separate)
                    if 'original_results' in bias_results:
                        bias_results_for_logging = bias_results.get('original_results', {})
                        if 'mitigated_results' in bias_results:
                            logger.info("Bias mitigation completed. Check comparison report for details.")
                        else:
                            logger.warning("Bias mitigation was attempted but no mitigated results were generated.")
                            if 'error' in bias_results:
                                logger.error(f"Mitigation error: {bias_results.get('error')}")
                    else:
                        bias_results_for_logging = bias_results
                else:
                    logger.info("Running bias detection only (mitigation disabled)...")
                    bias_results = bias_detector.detect_bias(
                        split='test',
                        max_samples=max_samples
                    )
                    bias_results_for_logging = bias_results
                
                # Log bias metrics to MLflow
                if bias_results_for_logging and 'overall_performance' in bias_results_for_logging:
                    mlflow.log_metric("bias_detected", 1 if bias_results_for_logging.get('bias_detected', False) else 0)
                    
                    # Log slice performance differences and fairness metrics (Fairlearn-based)
                    for feature, feature_data in bias_results_for_logging.get('slices', {}).items():
                        # Get group metrics (Fairlearn structure)
                        group_metrics = feature_data.get('group_metrics', {})
                        fairness_metrics = feature_data.get('fairness_metrics', {})
                        
                        if len(group_metrics) >= 2:
                            # Log performance differences
                            accuracies = [m['accuracy'] for m in group_metrics.values()]
                            perf_diff = max(accuracies) - min(accuracies)
                            mlflow.log_metric(f"bias_performance_diff_{feature}", perf_diff)
                            
                            # Log Fairlearn fairness metrics
                            if 'demographic_parity_difference' in fairness_metrics:
                                mlflow.log_metric(f"bias_dp_diff_{feature}", 
                                                 abs(fairness_metrics['demographic_parity_difference']))
                            if 'demographic_parity_ratio' in fairness_metrics:
                                mlflow.log_metric(f"bias_dp_ratio_{feature}", 
                                                 fairness_metrics['demographic_parity_ratio'])
                            if 'equalized_odds_difference' in fairness_metrics:
                                mlflow.log_metric(f"bias_eo_diff_{feature}", 
                                                 abs(fairness_metrics['equalized_odds_difference']))
                            if 'equalized_odds_ratio' in fairness_metrics:
                                mlflow.log_metric(f"bias_eo_ratio_{feature}", 
                                                 fairness_metrics['equalized_odds_ratio'])
                
                # If mitigation was applied, also log mitigated results
                if mitigation_enabled and isinstance(bias_results, dict) and 'mitigated_results' in bias_results:
                    mlflow.log_metric("bias_mitigation_applied", 1)
                    mitigated_results = bias_results.get('mitigated_results', {})
                    if 'overall_performance' in mitigated_results:
                        mlflow.log_metric("bias_mitigated_accuracy", 
                                         mitigated_results['overall_performance'].get('accuracy', 0))
                        mlflow.log_metric("bias_mitigated_bias_detected", 
                                         1 if mitigated_results.get('bias_detected', False) else 0)
                
                # Log bias report as artifact (look in dataset-specific subdirectory)
                bias_reports_dir = metadata_dir / "bias_reports" / dataset_name
                if bias_reports_dir.exists():
                    bias_report_files = list(bias_reports_dir.glob(f"bias_report_{dataset_name}_*.json"))
                    if bias_report_files:
                        mlflow.log_artifact(str(bias_report_files[-1]), f"bias_reports/{dataset_name}")
                    
                    bias_html_files = list(bias_reports_dir.glob(f"bias_report_{dataset_name}_*.html"))
                    if bias_html_files:
                        mlflow.log_artifact(str(bias_html_files[-1]), f"bias_reports/{dataset_name}")
                    
                    # Log mitigation comparison report if available
                    if mitigation_enabled:
                        comparison_files = list(bias_reports_dir.glob(f"bias_mitigation_comparison_*.html"))
                        if comparison_files:
                            mlflow.log_artifact(str(comparison_files[-1]), f"bias_reports/{dataset_name}")
                else:
                    # Fallback to old location (for backwards compatibility)
                    bias_report_files = list((metadata_dir / "bias_reports").glob(f"bias_report_{dataset_name}_*.json"))
                    if bias_report_files:
                        mlflow.log_artifact(str(bias_report_files[-1]), f"bias_reports/{dataset_name}")
                    
                    bias_html_files = list((metadata_dir / "bias_reports").glob(f"bias_report_{dataset_name}_*.html"))
                    if bias_html_files:
                        mlflow.log_artifact(str(bias_html_files[-1]), f"bias_reports/{dataset_name}")
                
                logger.info("Bias detection completed successfully.")
                
            except Exception as e:
                logger.error(f"Error during bias detection: {e}", exc_info=True)
        elif bias_config.get('enabled', False) and not BIAS_DETECTION_AVAILABLE:
            logger.warning("Bias detection is enabled but module is not available.")
        
        # Model Interpretability (SHAP and LIME)
        interpretability_config = config.get('interpretability', {})
        if interpretability_config.get('enabled', False) and INTERPRETABILITY_AVAILABLE:
            logger.info("\n" + "="*60)
            logger.info("Running Model Interpretability Analysis (SHAP & LIME)...")
            logger.info("="*60)
            
            try:
                # Get data path for interpretability (same as bias detection)
                # For sampled data, the sampled path is the test directory itself
                if hasattr(test_gen, '_sampled_path') and test_gen._sampled_path:
                    # The sampled path is already the test directory (e.g., /tmp/data_subset_xxx/test)
                    interpretability_data_path = Path(test_gen._sampled_path)
                    logger.info(f"Using sampled test directory for interpretability: {interpretability_data_path}")
                else:
                    # Use original data path (latest_day is already the full path)
                    interpretability_data_path = latest_day
                    logger.info(f"Using original data path for interpretability: {interpretability_data_path}")
                
                # Create interpreter
                interpreter = ModelInterpreter(
                    model=trainer.model,
                    class_names=class_names,
                    output_dir=metadata_dir / "interpretability_reports",
                    image_size=image_size
                )
                
                # Generate interpretability report
                max_samples = 64 if dry_run else interpretability_config.get('max_samples', 10)
                shap_background = interpretability_config.get('shap_background_samples', 50)
                shap_evals = interpretability_config.get('shap_max_evals', 100)
                lime_explanations = interpretability_config.get('lime_num_explanations', 5)
                
                interpretability_results = interpreter.generate_interpretability_report(
                    data_path=interpretability_data_path,
                    split='test',
                    max_samples=max_samples,
                    shap_background_samples=shap_background,
                    shap_max_evals=shap_evals,
                    lime_num_explanations=lime_explanations
                )
                
                # Log interpretability metrics to MLflow
                if interpretability_results and 'error' not in interpretability_results:
                    mlflow.log_metric("interpretability_images_analyzed", 
                                     interpretability_results.get('num_images_loaded', 0))
                    
                    # Log SHAP results
                    if 'shap' in interpretability_results and interpretability_results['shap'].get('success'):
                        mlflow.log_metric("shap_explanations_generated", 
                                         interpretability_results['shap'].get('num_explained', 0))
                        mlflow.log_metric("shap_background_samples", 
                                         interpretability_results['shap'].get('background_samples', 0))
                    
                    # Log LIME results
                    if 'lime' in interpretability_results and interpretability_results['lime'].get('success'):
                        mlflow.log_metric("lime_explanations_generated", 
                                         interpretability_results['lime'].get('num_explained', 0))
                
                # Log interpretability artifacts
                interpretability_reports_dir = metadata_dir / "interpretability_reports"
                if interpretability_reports_dir.exists():
                    # Log JSON report
                    report_file = interpretability_reports_dir / "interpretability_report.json"
                    if report_file.exists():
                        mlflow.log_artifact(str(report_file), "interpretability_reports")
                    
                    # Log SHAP plots
                    if 'shap' in interpretability_results and 'plots' in interpretability_results['shap']:
                        for plot_path in interpretability_results['shap']['plots']:
                            if Path(plot_path).exists():
                                mlflow.log_artifact(plot_path, "interpretability_reports/shap")
                    
                    # Log LIME plots
                    if 'lime' in interpretability_results and 'plots' in interpretability_results['lime']:
                        for plot_path in interpretability_results['lime']['plots']:
                            if Path(plot_path).exists():
                                mlflow.log_artifact(plot_path, "interpretability_reports/lime")
                
                logger.info("Model interpretability analysis completed successfully.")
                
            except Exception as e:
                logger.error(f"Error during interpretability analysis: {e}", exc_info=True)
        elif interpretability_config.get('enabled', False) and not INTERPRETABILITY_AVAILABLE:
            logger.warning("Interpretability is enabled but module is not available.")
        
        # Log training history
        for epoch, (loss, acc, val_loss, val_acc) in enumerate(
            zip(history['loss'], history['accuracy'],
                history['val_loss'], history['val_accuracy']), 1
        ):
            mlflow.log_metric("train_loss", loss, step=epoch)
            mlflow.log_metric("train_accuracy", acc, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
        
        # Save model
        trainer.save_model()
        
        # Log model artifact with input example for signature inference
        # Create a sample input with the model's expected input shape
        input_example = np.random.rand(1, *input_shape).astype(np.float32)
        mlflow.tensorflow.log_model(
            trainer.model,
            name=f"{model_name}_model",
            registered_model_name=f"{dataset_name}_{model_name}",
            input_example=input_example
        )
    
    # Save metadata
    model_info = {
        'model': model_name,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat(),
        'dataset': dataset_name,
        'num_classes': num_classes,
        'epochs': epochs,
        'batch_size': batch_size,
        'image_size': image_size,
        'train_samples': train_gen.samples,
        'val_samples': val_gen.samples,
        'test_samples': test_gen.samples,
        'hyperparameter_tuning': tuning_enabled
    }
    
    # Write metadata
    metadata_file = metadata_dir / "training_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(model_info, f, indent=2)
    logger.info(f"Metadata saved to {metadata_file}")
    
    # Save model info with model path
    model_path = models_dir / f"{model_name}_best.keras"
    model_info['model_path'] = str(model_path)
    # Create relative path with date partition structure including dataset and model name
    model_info['model_relative_path'] = f"models/{year}/{month}/{day}/{timestamp}/{dataset_name}_{model_name}/{model_name}_best.keras"
    model_info['dry_run'] = dry_run
    
    summary_file = metadata_dir / "training_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(model_info, f, indent=2)
    logger.info(f"Training summary saved to {summary_file}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Custom CNN training completed for {dataset_name}")
    logger.info(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Metadata saved to: {metadata_dir}")
    logger.info(f"{'='*60}\n")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Custom CNN model for medical image classification')
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config YAML file. Default: ModelDevelopment/config/vision_training.yml'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default=None,
        help='Path to preprocessed data (overrides config)'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Base path for outputs (overrides config)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (overrides config)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Batch size for training (overrides config)'
    )
    parser.add_argument(
        '--image_size',
        type=int,
        nargs=2,
        default=None,
        help='Image size (height width) (overrides config)'
    )
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        default=None,
        help='Datasets to train (overrides config)'
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Dry run mode: Use only 64 images for quick testing'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = ConfigLoader(args.config)
    
    # Override config with command line arguments if provided
    data_path = args.data_path or config.get('paths.data_path', '../../DataPipeline/data/preprocessed')
    output_path = args.output_path or config.get('paths.output_path', '../data')
    epochs = args.epochs
    batch_size = args.batch_size
    image_size = tuple(args.image_size) if args.image_size else None
    datasets = args.datasets or config.get('training.datasets', ['tb', 'lung_cancer_ct_scan'])
    
    # Train models for each dataset
    for dataset_name in datasets:
        try:
            train_dataset(
                dataset_name=dataset_name,
                base_data_path=data_path,
                output_base_path=output_path,
                config=config,
                epochs=epochs,
                batch_size=batch_size,
                image_size=image_size,
                dry_run=args.dry_run
            )
        except Exception as e:
            logger.error(f"Error training {dataset_name}: {e}", exc_info=True)
            continue
    
    logger.info("Custom CNN training completed!")


if __name__ == "__main__":
    main()

