#!/usr/bin/env python3
"""
Model Selection Script for Vision Models

Compares ResNet18, ViT, and Custom CNN models and selects the best one
based on test accuracy (or configurable metric).
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_model_metadata(output_path: Path, dataset: str) -> List[Dict]:
    """
    Find all training_metadata.json files for a given dataset.
    
    Args:
        output_path: Base output path (e.g., /workspace/models)
        dataset: Dataset name (tb or lung_cancer_ct_scan)
        
    Returns:
        List of metadata dictionaries with model info
    """
    metadata_files = []
    
    # Search for training_metadata.json files
    # Models are saved in date-partitioned structure: YYYY/MM/DD/HHMMSS/
    search_paths = [
        output_path / dataset,  # Direct dataset folder
        output_path,  # Root models folder
    ]
    
    for search_path in search_paths:
        if not search_path.exists():
            continue
            
        # Find all training_metadata.json files
        for metadata_file in search_path.rglob("training_metadata.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    
                # Verify this is for the correct dataset
                if metadata.get('dataset') == dataset:
                    # Find corresponding model file
                    model_name = metadata.get('model', 'Unknown')
                    model_dir = metadata_file.parent
                    
                    # Look for model file
                    model_file = None
                    for pattern in [f"{model_name}_best.keras", f"{model_name}.keras", "*.keras"]:
                        matches = list(model_dir.glob(pattern))
                        if matches:
                            model_file = matches[0]
                            break
                    
                    # Also check in models subdirectory
                    if model_file is None:
                        models_dir = output_path / "models" / metadata_file.parent.relative_to(output_path)
                        for pattern in [f"{model_name}_best.keras", f"{model_name}.keras"]:
                            matches = list(models_dir.glob(pattern))
                            if matches:
                                model_file = matches[0]
                                break
                    
                    metadata['metadata_path'] = str(metadata_file)
                    metadata['model_path'] = str(model_file) if model_file else None
                    metadata_files.append(metadata)
                    
            except Exception as e:
                logger.warning(f"Error reading {metadata_file}: {e}")
                continue
    
    return metadata_files


def select_best_model(
    metadata_list: List[Dict],
    metric: str = 'test_accuracy'
) -> Tuple[Optional[Dict], List[Dict]]:
    """
    Select the best model based on a metric.
    
    Args:
        metadata_list: List of model metadata dictionaries
        metric: Metric to use for selection (default: 'test_accuracy')
        
    Returns:
        Tuple of (best_model_metadata, all_models_sorted)
    """
    if not metadata_list:
        logger.error("No model metadata found!")
        return None, []
    
    # Extract metric values
    models_with_metrics = []
    for metadata in metadata_list:
        metrics = metadata.get('metrics', {})
        metric_value = metrics.get(metric, 0.0)
        
        if metric_value is None:
            logger.warning(f"Model {metadata.get('model')} has no {metric} value")
            continue
            
        models_with_metrics.append({
            'metadata': metadata,
            'metric_value': float(metric_value),
            'model_name': metadata.get('model', 'Unknown')
        })
    
    if not models_with_metrics:
        logger.error(f"No models with {metric} metric found!")
        return None, []
    
    # Sort by metric value (descending)
    models_with_metrics.sort(key=lambda x: x['metric_value'], reverse=True)
    
    best = models_with_metrics[0]
    logger.info(f"Best model: {best['model_name']} with {metric}={best['metric_value']:.4f}")
    
    # Log all models
    logger.info("\nModel Comparison:")
    logger.info("-" * 60)
    for i, model_info in enumerate(models_with_metrics, 1):
        marker = "★ BEST" if i == 1 else ""
        logger.info(f"{i}. {model_info['model_name']:20s} {metric}={model_info['metric_value']:.4f} {marker}")
    logger.info("-" * 60)
    
    return best['metadata'], [m['metadata'] for m in models_with_metrics]


def save_selection_result(
    best_model: Dict,
    all_models: List[Dict],
    output_path: Path,
    dataset: str
):
    """
    Save model selection results to JSON file.
    
    Args:
        best_model: Best model metadata
        all_models: All models sorted by performance
        output_path: Output directory
        dataset: Dataset name
    """
    selection_result = {
        'timestamp': datetime.now().isoformat(),
        'dataset': dataset,
        'best_model': {
            'model_name': best_model.get('model'),
            'model_path': best_model.get('model_path'),
            'metadata_path': best_model.get('metadata_path'),
            'metrics': best_model.get('metrics', {}),
            'test_accuracy': best_model.get('metrics', {}).get('test_accuracy', 0.0)
        },
        'all_models': [
            {
                'model_name': m.get('model'),
                'model_path': m.get('model_path'),
                'metrics': m.get('metrics', {}),
                'test_accuracy': m.get('metrics', {}).get('test_accuracy', 0.0)
            }
            for m in all_models
        ],
        'selection_metric': 'test_accuracy'
    }
    
    # Save to output path
    output_file = output_path / "model_selection_result.json"
    with open(output_file, 'w') as f:
        json.dump(selection_result, f, indent=2)
    
    logger.info(f"Selection result saved to: {output_file}")
    
    # Also save best model info to a simple file for Cloud Build
    best_info_file = output_path / "best_model_info.json"
    with open(best_info_file, 'w') as f:
        json.dump(selection_result['best_model'], f, indent=2)
    
    logger.info(f"Best model info saved to: {best_info_file}")


def main():
    parser = argparse.ArgumentParser(description='Select best Vision model from trained models')
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='Base output path where models are stored (e.g., /workspace/models)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Dataset name (tb or lung_cancer_ct_scan)'
    )
    parser.add_argument(
        '--metric',
        type=str,
        default='test_accuracy',
        help='Metric to use for selection (default: test_accuracy)'
    )
    
    args = parser.parse_args()
    
    output_path = Path(args.output_path)
    if not output_path.exists():
        logger.error(f"Output path does not exist: {output_path}")
        return 1
    
    logger.info(f"Searching for models in: {output_path}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Selection metric: {args.metric}")
    
    # Find all model metadata
    metadata_list = find_model_metadata(output_path, args.dataset)
    
    if not metadata_list:
        logger.error(f"No model metadata found for dataset: {args.dataset}")
        logger.info("Expected files: training_metadata.json in date-partitioned directories")
        return 1
    
    logger.info(f"Found {len(metadata_list)} model(s)")
    
    # Select best model
    best_model, all_models = select_best_model(metadata_list, args.metric)
    
    if best_model is None:
        logger.error("Failed to select best model")
        return 1
    
    # Save selection result
    save_selection_result(best_model, all_models, output_path, args.dataset)
    
    logger.info("\n" + "="*60)
    logger.info("Model Selection Complete!")
    logger.info(f"Best Model: {best_model.get('model')}")
    logger.info(f"Test Accuracy: {best_model.get('metrics', {}).get('test_accuracy', 0.0):.4f}")
    logger.info(f"Model Path: {best_model.get('model_path')}")
    logger.info("="*60)
    
    return 0


if __name__ == "__main__":
    exit(main())

