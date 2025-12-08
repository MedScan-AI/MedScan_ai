"""
Simplified training script for ResNet18 model.
Trains ResNet18 for Tuberculosis (TB) and Lung Cancer detection.
Only does basic training and saving - no mlflow, bias detection, hyperparameter tuning, etc.
"""

import os
import sys
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import argparse

import numpy as np
import yaml
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)


class ConfigLoader:
    """Load and manage configuration from YAML file."""
    
    def __init__(self, config_path: str):
        """Initialize ConfigLoader."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        logger.info(f"Loaded configuration from {self.config_path}")
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation."""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value


class DataLoader:
    """Load preprocessed image data."""
    
    def __init__(self, base_path: str, dataset_name: str, config: ConfigLoader):
        """Initialize DataLoader."""
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
        logger.info(f"Found years: {[y.name for y in sorted(years, key=lambda x: int(x.name))]}, using latest: {latest_year.name}")
        
        # Find latest month
        months = [d for d in latest_year.iterdir() if d.is_dir() and d.name.isdigit()]
        if not months:
            raise ValueError(f"No month directories found in {latest_year}")
        latest_month = max(months, key=lambda x: int(x.name))
        logger.info(f"Found months in {latest_year.name}: {[m.name for m in sorted(months, key=lambda x: int(x.name))]}, using latest: {latest_month.name}")
        
        # Find latest day
        days = [d for d in latest_month.iterdir() if d.is_dir() and d.name.isdigit()]
        if not days:
            raise ValueError(f"No day directories found in {latest_month}")
        latest_day = max(days, key=lambda x: int(x.name))
        logger.info(f"Found days in {latest_year.name}/{latest_month.name}: {[d.name for d in sorted(days, key=lambda x: int(x.name))]}, using latest: {latest_day.name}")
        
        logger.info(f"Selected latest partition: {latest_day}")
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
    
    def create_data_generators(
        self,
        batch_size: int = 32,
        image_size: Tuple[int, int] = (224, 224),
        validation_split: float = None
    ) -> Tuple[ImageDataGenerator, ImageDataGenerator, ImageDataGenerator]:
        """Create data generators for train, validation, and test sets."""
        if validation_split is None:
            validation_split = self.config.get('training.validation_split', 0.2)
        
        # No data augmentation - only rescale pixel values
        train_datagen = ImageDataGenerator(
            rescale=1.0/255.0,
            validation_split=validation_split
        )
        
        # No augmentation for validation and test
        val_test_datagen = ImageDataGenerator(
            rescale=1.0/255.0,
            validation_split=validation_split
        )
        
        train_path = self.data_path / "train"
        test_path = self.data_path / "test"
        val_path = self.data_path / "valid" if (self.data_path / "valid").exists() else None
        
        # Get classes
        classes = [d.name for d in train_path.iterdir() if d.is_dir()]
        classes.sort()
        logger.info(f"Found classes: {classes}")
        
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
        
        # Validation generator
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
        
        logger.info(f"Train samples: {train_gen.samples}")
        logger.info(f"Validation samples: {val_gen.samples}")
        logger.info(f"Test samples: {test_gen.samples}")
        
        return train_gen, val_gen, test_gen


def find_latest_model(output_base_path: str, dataset_name: str, model_name: str) -> Optional[Path]:
    """
    Find the latest saved model for a given dataset.
    
    Args:
        output_base_path: Base path for model outputs
        dataset_name: Name of dataset ('tb' or 'lung_cancer_ct_scan')
        model_name: Name of the model architecture
        
    Returns:
        Path to the latest model file, or None if no model found
    """
    data_dir = Path(output_base_path)
    models_base_dir = data_dir / "models"
    
    if not models_base_dir.exists():
        return None
    
    # Find all year directories
    years = [d for d in models_base_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    if not years:
        return None
    
    latest_model_path = None
    latest_timestamp = None
    
    # Search through all year/month/day/timestamp directories
    for year_dir in years:
        months = [d for d in year_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        for month_dir in months:
            days = [d for d in month_dir.iterdir() if d.is_dir() and d.name.isdigit()]
            for day_dir in days:
                timestamps = [d for d in day_dir.iterdir() if d.is_dir() and d.name.isdigit()]
                for timestamp_dir in timestamps:
                    # Check if this directory contains the model for this dataset
                    model_dir = timestamp_dir / f"{dataset_name}_{model_name}"
                    model_file = model_dir / f"{model_name}_best.keras"
                    
                    if model_file.exists():
                        # Create a comparable timestamp (YYYYMMDDHHMMSS)
                        timestamp_str = f"{year_dir.name}{month_dir.name}{day_dir.name}{timestamp_dir.name}"
                        if latest_timestamp is None or timestamp_str > latest_timestamp:
                            latest_timestamp = timestamp_str
                            latest_model_path = model_file
    
    return latest_model_path


class ModelBuilder:
    """Build ResNet18 model architecture."""
    
    def __init__(self, config: ConfigLoader):
        """Initialize ModelBuilder with configuration."""
        self.config = config
    
    def _residual_block(self, x, filters, stride=1, downsample=False):
        """Residual block for ResNet18."""
        identity = x
        
        # First conv layer
        x = layers.Conv2D(filters, kernel_size=3, strides=stride, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        # Second conv layer
        x = layers.Conv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        
        # Downsample identity if needed
        if downsample:
            identity = layers.Conv2D(filters, kernel_size=1, strides=stride, padding='same', use_bias=False)(identity)
            identity = layers.BatchNormalization()(identity)
        
        # Add skip connection
        x = layers.Add()([x, identity])
        x = layers.ReLU()(x)
        
        return x
    
    def build_cnn_resnet18(
        self,
        input_shape: Tuple[int, int, int],
        num_classes: int
    ) -> models.Model:
        """
        Build CNN model based on ResNet18 with full training (no transfer learning).
        ResNet18 has ~11.69M parameters.
        
        Args:
            input_shape: Input image shape (height, width, channels)
            num_classes: Number of output classes
            
        Returns:
            Compiled Keras model
        """
        resnet_config = self.config.get('models.resnet18', {})
        
        # Input layer
        inputs = keras.Input(shape=input_shape)
        
        # Initial conv layer: 7x7, 64 filters, stride 2
        x = layers.Conv2D(64, kernel_size=7, strides=2, padding='same', use_bias=False)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        # Max pooling: 3x3, stride 2
        x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
        
        # Residual blocks
        # Block 1: 2 layers, 64 filters
        x = self._residual_block(x, filters=64, stride=1, downsample=False)
        x = self._residual_block(x, filters=64, stride=1, downsample=False)
        
        # Block 2: 2 layers, 128 filters
        x = self._residual_block(x, filters=128, stride=2, downsample=True)
        x = self._residual_block(x, filters=128, stride=1, downsample=False)
        
        # Block 3: 2 layers, 256 filters
        x = self._residual_block(x, filters=256, stride=2, downsample=True)
        x = self._residual_block(x, filters=256, stride=1, downsample=False)
        
        # Block 4: 2 layers, 512 filters
        x = self._residual_block(x, filters=512, stride=2, downsample=True)
        x = self._residual_block(x, filters=512, stride=1, downsample=False)
        
        # Global average pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Add custom classification head
        head_config = resnet_config.get('head', {})
        x = layers.Dropout(head_config.get('dropout_1', 0.5))(x)
        x = layers.Dense(head_config.get('dense_1', 512), activation=head_config.get('activation', 'relu'))(x)
        x = layers.Dropout(head_config.get('dropout_2', 0.3))(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = models.Model(inputs, outputs)
        
        learning_rate = resnet_config.get('learning_rate', 0.01)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model


def train_dataset(
    dataset_name: str,
    base_data_path: str,
    output_base_path: str,
    config: ConfigLoader,
    epochs: int = None,
    batch_size: int = None,
    image_size: Tuple[int, int] = None,
    load_latest: bool = False
):
    """
    Train ResNet18 model for a specific dataset.
    
    Args:
        dataset_name: Name of dataset ('tb' or 'lung_cancer_ct_scan')
        base_data_path: Base path to preprocessed data
        output_base_path: Base path for model outputs
        config: ConfigLoader instance
        epochs: Number of training epochs (from config if None)
        batch_size: Batch size for training (from config if None)
        image_size: Target image size (from config if None)
    """
    logger.info(f"Starting ResNet18 training for {dataset_name} dataset")
    
    # Get training config
    training_config = config.get('training', {})
    if epochs is None:
        epochs = training_config.get('epochs', 50)
    if batch_size is None:
        batch_size = training_config.get('batch_size', 32)
    if image_size is None:
        image_size = tuple(training_config.get('image_size', [224, 224]))
    
    # Setup directory structure with date partitions
    now = datetime.now()
    year = now.strftime("%Y")
    month = now.strftime("%m")
    day = now.strftime("%d")
    timestamp = now.strftime("%H%M%S")
    
    model_name = 'CNN_ResNet18'
    
    data_dir = Path(output_base_path)
    models_dir = data_dir / "models" / year / month / day / timestamp / f"{dataset_name}_{model_name}"
    
    # Create directories
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data_loader = DataLoader(base_data_path, dataset_name, config)
    train_gen, val_gen, test_gen = data_loader.create_data_generators(
        batch_size=batch_size,
        image_size=image_size
    )
    
    # Get number of classes and class names
    class_names = data_loader.get_classes()
    num_classes = len(class_names)
    input_shape = (*image_size, 3)
    
    # Load existing model or build new one
    if load_latest:
        latest_model_path = find_latest_model(output_base_path, dataset_name, model_name)
        if latest_model_path and latest_model_path.exists():
            logger.info(f"Loading latest model from: {latest_model_path}")
            model = keras.models.load_model(str(latest_model_path))
            logger.info("Model loaded successfully. Continuing training...")
        else:
            logger.warning(f"No existing model found for {dataset_name}. Building new model...")
            model_builder = ModelBuilder(config)
            model = model_builder.build_cnn_resnet18(input_shape, num_classes)
    else:
        # Build model
        model_builder = ModelBuilder(config)
        model = model_builder.build_cnn_resnet18(input_shape, num_classes)
    
    # Print model parameters
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
    
    # Train model
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {model_name}")
    logger.info(f"{'='*60}\n")
    
    # Calculate steps per epoch for better performance
    steps_per_epoch = (train_gen.samples + train_gen.batch_size - 1) // train_gen.batch_size
    validation_steps = (val_gen.samples + val_gen.batch_size - 1) // val_gen.batch_size
    
    history = model.fit(
        train_gen,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps,
        verbose=1
    )
    
    # Evaluate model
    logger.info(f"Evaluating {model_name} on test set...")
    test_steps = (test_gen.samples + test_gen.batch_size - 1) // test_gen.batch_size
    test_results = model.evaluate(test_gen, steps=test_steps, verbose=1)
    
    metrics = {
        'test_loss': test_results[0],
        'test_accuracy': test_results[1]
    }
    
    logger.info(f"Test Loss: {metrics['test_loss']:.4f}")
    logger.info(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    
    # Save model (same naming convention as reference script)
    model_path = models_dir / f"{model_name}_best.keras"
    model.save(str(model_path))
    logger.info(f"Model saved to {model_path}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"ResNet18 training completed for {dataset_name}")
    logger.info(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"{'='*60}\n")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train ResNet18 model for medical image classification')
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config YAML file. Default: ModelDevelopment/Vision/rd/config.yml'
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
        '--load_latest',
        action='store_true',
        help='Load the latest saved model and continue training'
    )
    
    args = parser.parse_args()
    
    # Determine config path
    if args.config is None:
        script_dir = Path(__file__).parent
        config_path = script_dir / "config.yml"
    else:
        config_path = Path(args.config)
    
    # Load configuration
    config = ConfigLoader(str(config_path))
    
    # Override config with command line arguments if provided
    data_path = args.data_path or config.get('paths.data_path', '../../DataPipeline/data/preprocessed')
    output_path = args.output_path or config.get('paths.output_path', '../../data')
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
                load_latest=args.load_latest
            )
        except Exception as e:
            logger.error(f"Error training {dataset_name}: {e}", exc_info=True)
            continue
    
    logger.info("ResNet18 training completed!")


if __name__ == "__main__":
    main()
