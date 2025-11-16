"""
validate.py - Validate trained Vision models
"""
import logging
import json
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelValidator:
    """Validate trained models"""
    
    def __init__(
        self,
        model_path: str,
        test_data_path: str,
        image_size: tuple = (224, 224),
        batch_size: int = 32
    ):
        self.model_path = Path(model_path)
        self.test_data_path = Path(test_data_path)
        self.image_size = image_size
        self.batch_size = batch_size
        
        # Load model
        logger.info(f"Loading model from {self.model_path}")
        self.model = tf.keras.models.load_model(str(self.model_path))
        
        # Create test data generator
        self.test_datagen = ImageDataGenerator(rescale=1.0/255.0)
        self.test_generator = self.test_datagen.flow_from_directory(
            str(self.test_data_path),
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
    
    def validate(self) -> dict:
        """
        Validate model performance.
        
        Returns:
            Dict with validation metrics
        """
        logger.info("Starting model validation...")
        
        # Evaluate
        results = self.model.evaluate(self.test_generator, verbose=1)
        
        # Get metric names
        metric_names = self.model.metrics_names
        
        # Create results dict
        validation_results = {
            name: float(value)
            for name, value in zip(metric_names, results)
        }
        
        logger.info("Validation Results:")
        for name, value in validation_results.items():
            logger.info(f"  {name}: {value:.4f}")
        
        # Check thresholds
        # Keras 3 may return 'compile_metrics' or 'accuracy' depending on version
        accuracy = validation_results.get('accuracy') or validation_results.get('compile_metrics') or 0.0
        passed = accuracy >= 0.7  # 70% accuracy threshold
        
        validation_results['passed'] = passed
        validation_results['threshold'] = 0.7
        
        if passed:
            logger.info(f"✓ Validation PASSED (accuracy: {accuracy:.4f} >= 0.7)")
        else:
            logger.error(f"✗ Validation FAILED (accuracy: {accuracy:.4f} < 0.7)")
        
        return validation_results
    
    def save_results(self, results: dict, output_path: str):
        """Save validation results to JSON"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")


def main():
    """Main validation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate Vision model')
    parser.add_argument('--model_path', required=True, help='Path to .keras model')
    parser.add_argument('--test_data_path', required=True, help='Path to test data')
    parser.add_argument('--output_file', default='validation_results.json')
    
    args = parser.parse_args()
    
    validator = ModelValidator(
        model_path=args.model_path,
        test_data_path=args.test_data_path
    )
    
    results = validator.validate()
    validator.save_results(results, args.output_file)
    
    # Exit with non-zero if validation failed
    if not results['passed']:
        exit(1)


if __name__ == "__main__":
    main()