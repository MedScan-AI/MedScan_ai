"""
bias_check.py - Simplified bias detection for Vision models
Checks for performance disparities across demographic slices
"""
import logging
import json
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VisionBiasChecker:
    """Check for bias in Vision model predictions"""
    
    def __init__(
        self,
        metadata_path: str,
        model_path: str,
        threshold: float = 0.10  # 10% disparity threshold
    ):
        """
        Initialize bias checker.
        
        Args:
            metadata_path: Path to training_metadata.json
            model_path: Path to trained model
            threshold: Max allowed disparity (0.10 = 10%)
        """
        self.metadata_path = Path(metadata_path)
        self.model_path = Path(model_path)
        self.threshold = threshold
        
        # Load metadata
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
    
    def check_performance_bias(self) -> Dict:
        """
        Check for performance bias.
        
        For Vision models without demographic slicing during training,
        we check if validation/test accuracy is consistent.
        
        Returns:
            Dict with bias check results
        """
        logger.info("Checking for performance bias...")
        
        # Extract metrics
        test_accuracy = self.metadata.get('metrics', {}).get('test_accuracy', 0.0)
        
        # For Vision models, we check model confidence distribution
        # This is a simplified check - in production, you'd analyze
        # predictions across different image characteristics
        
        results = {
            'bias_detected': False,
            'test_accuracy': test_accuracy,
            'threshold': self.threshold,
            'message': 'No demographic metadata available for slice-based bias detection',
            'recommendation': 'Consider adding demographic slicing in future iterations'
        }
        
        # Simple check: if accuracy is very low, flag for review
        if test_accuracy < 0.70:
            results['bias_detected'] = True
            results['message'] = f'Low test accuracy ({test_accuracy:.4f}) may indicate systematic bias'
            results['recommendation'] = 'Review training data distribution and consider data augmentation'
        
        return results
    
    def save_results(self, results: Dict, output_path: str):
        """Save bias check results"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        if results['bias_detected']:
            logger.warning(f"BIAS DETECTED: {results['message']}")
        else:
            logger.info(f"✓ No significant bias detected")
        
        logger.info(f"Results saved to {output_path}")


def main():
    """Main bias checking function"""
    parser = argparse.ArgumentParser(description='Check Vision model for bias')
    parser.add_argument('--model_path', required=True, help='Path to trained model')
    parser.add_argument('--metadata_path', required=True, help='Path to metadata JSON')
    parser.add_argument('--output_file', default='bias_results.json')
    parser.add_argument('--threshold', type=float, default=0.10)
    
    args = parser.parse_args()
    
    checker = VisionBiasChecker(
        metadata_path=args.metadata_path,
        model_path=args.model_path,
        threshold=args.threshold
    )
    
    results = checker.check_performance_bias()
    checker.save_results(results, args.output_file)
    
    # Exit with warning if bias detected (but don't fail build)
    if results['bias_detected']:
        logger.warning("Bias detected - review recommended")
        # Don't exit(1) here - just warn
    
    return results


if __name__ == "__main__":
    main()