"""
Interpretability module for model explanations using SHAP and LIME.
Provides SHAP and LIME explanations for image classification models.
"""

import os
import json
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import base64
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

logger = logging.getLogger(__name__)

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
    logger.info("SHAP successfully imported")
except ImportError as e:
    SHAP_AVAILABLE = False
    logger.warning(f"SHAP not available. SHAP explanations will be skipped. Error: {e}")

# Try to import LIME
try:
    from lime import lime_image
    from skimage.segmentation import mark_boundaries
    LIME_AVAILABLE = True
    logger.info("LIME successfully imported")
except ImportError as e:
    LIME_AVAILABLE = False
    logger.warning(f"LIME not available. LIME explanations will be skipped. Error: {e}")


class ModelInterpreter:
    """Generate SHAP and LIME explanations for image classification models."""
    
    def __init__(
        self,
        model: Any,
        class_names: List[str],
        output_dir: Path,
        image_size: Tuple[int, int] = (224, 224)
    ):
        """
        Initialize ModelInterpreter.
        
        Args:
            model: Trained Keras model
            class_names: List of class names
            output_dir: Directory to save interpretability reports
            image_size: Input image size (height, width)
        """
        self.model = model
        self.class_names = class_names
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Suppress SHAP and Keras warnings
        if SHAP_AVAILABLE:
            warnings.filterwarnings('ignore', category=UserWarning, module='shap')
            warnings.filterwarnings('ignore', category=UserWarning, module='keras')
            # Suppress Keras functional model input structure warnings
            warnings.filterwarnings('ignore', message='.*structure of `inputs`.*')
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input."""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Resize if needed
        if image.shape[:2] != self.image_size:
            from PIL import Image as PILImage
            if len(image.shape) == 2:
                image = PILImage.fromarray(image).convert('RGB')
            else:
                image = PILImage.fromarray(image)
            image = image.resize(self.image_size)
            image = np.array(image)
        
        # Normalize to [0, 1] if needed
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        
        # Ensure shape is (height, width, channels)
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        
        return image
    
    def _load_sample_images(
        self,
        data_path: Path,
        split: str = 'test',
        max_samples: int = 10
    ) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """
        Load sample images for interpretability analysis.
        
        Args:
            data_path: Path to data directory (can be split directory or parent directory)
            split: Data split ('train', 'valid', 'test')
            max_samples: Maximum number of samples to load
            
        Returns:
            Tuple of (images, file_paths, true_labels)
        """
        images = []
        file_paths = []
        true_labels = []
        
        # Handle both cases (same as bias detection):
        # 1. data_path is a split directory (e.g., /tmp/data_subset_xxx/test) - use directly
        # 2. data_path is a parent directory (e.g., /app/data/preprocessed/tb/2025/10/24) - append split
        # Check if data_path is already a split directory by checking if it contains class subdirectories
        # and doesn't have train/test/valid as subdirectories
        has_split_subdirs = (data_path / 'train').exists() or (data_path / 'test').exists() or (data_path / 'valid').exists()
        has_class_dirs = any(d.is_dir() for d in data_path.iterdir() if d.name not in ['train', 'test', 'valid', 'val'])
        
        if has_class_dirs and not has_split_subdirs:
            # data_path is already a split directory (contains class directories directly)
            split_path = data_path
            logger.info(f"Using data_path as split directory: {split_path}")
        elif (data_path / split).exists():
            # data_path is parent directory, split subdirectory exists
            split_path = data_path / split
            logger.info(f"Using split subdirectory: {split_path}")
        else:
            # Try alternative split names
            split_path = None
            if split == 'test':
                for alt_path in ['test', 'val', 'valid']:
                    if (data_path / alt_path).exists():
                        split_path = data_path / alt_path
                        logger.info(f"Found alternative split path: {split_path}")
                        break
            
            if split_path is None:
                # Last resort: try data_path itself
                if has_class_dirs:
                    split_path = data_path
                    logger.info(f"Using data_path directly: {split_path}")
                else:
                    logger.error(f"Could not find {split} split at {data_path}")
                    return np.array([]), [], np.array([])
        
        if not split_path.exists():
            logger.error(f"Split path {split_path} does not exist")
            return np.array([]), [], np.array([])
        
        # Load images from class directories
        class_dirs = sorted([d for d in split_path.iterdir() if d.is_dir()])
        samples_per_class = max(1, max_samples // len(class_dirs)) if class_dirs else max_samples
        
        for class_dir in class_dirs:
            class_name = class_dir.name
            if class_name not in self.class_names:
                continue
            
            class_idx = self.class_names.index(class_name)
            image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpeg'))
            
            for img_file in image_files[:samples_per_class]:
                try:
                    img = Image.open(img_file)
                    img_array = self._preprocess_image(np.array(img))
                    images.append(img_array)
                    file_paths.append(str(img_file))
                    true_labels.append(class_idx)
                    
                    if len(images) >= max_samples:
                        break
                except Exception as e:
                    logger.warning(f"Failed to load image {img_file}: {e}")
                    continue
            
            if len(images) >= max_samples:
                break
        
        if len(images) == 0:
            logger.warning(f"No images loaded from {split_path}")
            return np.array([]), [], np.array([])
        
        images = np.array(images)
        true_labels = np.array(true_labels)
        
        logger.info(f"Loaded {len(images)} images for interpretability analysis")
        return images, file_paths, true_labels
    
    def generate_shap_explanations(
        self,
        images: np.ndarray,
        background_samples: int = 50,
        max_evals: int = 100
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanations for images.
        
        Args:
            images: Array of images to explain
            background_samples: Number of background samples for SHAP
            max_evals: Maximum evaluations for SHAP
            
        Returns:
            Dictionary containing SHAP results
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Skipping SHAP explanations.")
            return {'error': 'SHAP not available'}
        
        if len(images) == 0:
            logger.warning("No images provided for SHAP explanations.")
            return {'error': 'No images provided'}
        
        try:
            logger.info("Generating SHAP explanations...")
            
            # Prepare background data (subset of images)
            background_size = min(background_samples, len(images))
            background_indices = np.random.choice(len(images), background_size, replace=False)
            background = images[background_indices]
            
            # Explain a subset of images (SHAP can be slow)
            num_explain = min(5, len(images))
            explain_indices = np.random.choice(len(images), num_explain, replace=False)
            images_to_explain = images[explain_indices]
            
            logger.info(f"Explaining {num_explain} images with SHAP...")
            
            # Create a wrapper function for model prediction that ensures proper input format
            def model_predict_wrapper(x):
                """Wrapper to ensure inputs are in correct format for SHAP."""
                if isinstance(x, list):
                    x = np.array(x)
                # Ensure input is float32 and in correct shape
                if x.dtype != np.float32:
                    x = x.astype(np.float32)
                # Ensure values are in [0, 1] range
                if x.max() > 1.0:
                    x = x / 255.0
                return self.model.predict(x, verbose=0)
            
            # Try different SHAP explainers, starting with the most efficient
            shap_values_selected = None
            explainer_used = None
            
            # Method 1: Try GradientExplainer (fastest, but may not work with all models)
            try:
                logger.info("Trying GradientExplainer...")
                # Convert background to float32 if needed
                background_float = background.astype(np.float32) if background.dtype != np.float32 else background
                if background_float.max() > 1.0:
                    background_float = background_float / 255.0
                
                images_float = images_to_explain.astype(np.float32) if images_to_explain.dtype != np.float32 else images_to_explain
                if images_float.max() > 1.0:
                    images_float = images_float / 255.0
                
                # Suppress Keras warnings during SHAP execution
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=UserWarning, module='keras')
                    warnings.filterwarnings('ignore', message='.*structure of `inputs`.*')
                    warnings.filterwarnings('ignore', message='.*The structure.*')
                    explainer = shap.GradientExplainer(model_predict_wrapper, background_float)
                    shap_values = explainer.shap_values(images_float)
                
                # Handle multi-class output (shap_values is a list for each class)
                if isinstance(shap_values, list):
                    # For multi-class, use the predicted class's SHAP values
                    preds = self.model.predict(images_float, verbose=0)
                    pred_classes = np.argmax(preds, axis=1)
                    shap_values_selected = np.array([shap_values[cls][i] for i, cls in enumerate(pred_classes)])
                else:
                    shap_values_selected = shap_values
                
                explainer_used = "GradientExplainer"
                logger.info("GradientExplainer succeeded")
                    
            except Exception as e:
                logger.warning(f"GradientExplainer failed: {e}. Trying Partition explainer with masker...")
                
                # Method 2: Use Partition explainer with image masker (more compatible)
                try:
                    from shap.maskers import Image as ImageMasker
                    
                    # Prepare background for masker
                    background_uint8 = (background * 255).astype(np.uint8) if background.max() <= 1.0 else background.astype(np.uint8)
                    masker = ImageMasker(background_uint8)
                    
                    # Wrap model for masker (expects uint8)
                    def model_predict_uint8(x):
                        """Wrapper for masker that expects uint8 inputs."""
                        if isinstance(x, list):
                            x = np.array(x)
                        # Convert uint8 to float32 and normalize
                        x_float = x.astype(np.float32) / 255.0
                        return self.model.predict(x_float, verbose=0)
                    
                    images_uint8 = (images_to_explain * 255).astype(np.uint8) if images_to_explain.max() <= 1.0 else images_to_explain.astype(np.uint8)
                    
                    # Suppress Keras warnings during SHAP execution
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=UserWarning, module='keras')
                        warnings.filterwarnings('ignore', message='.*structure of `inputs`.*')
                        warnings.filterwarnings('ignore', message='.*The structure.*')
                        explainer = shap.Explainer(model_predict_uint8, masker, algorithm='partition')
                        shap_values = explainer(images_uint8, max_evals=min(max_evals, 200))
                    
                    if hasattr(shap_values, 'values'):
                        preds = self.model.predict(images_to_explain, verbose=0)
                        pred_classes = np.argmax(preds, axis=1)
                        # Extract values for predicted class if multi-dimensional
                        if len(shap_values.values.shape) > 3:
                            shap_values_selected = shap_values.values
                        else:
                            shap_values_selected = shap_values.values
                    else:
                        shap_values_selected = shap_values
                    
                    explainer_used = "PartitionExplainer"
                    logger.info("Partition explainer with masker succeeded")
                    
                except Exception as e2:
                    logger.warning(f"Partition explainer failed: {e2}. Trying DeepExplainer...")
                    
                    # Method 3: Try DeepExplainer (deprecated but sometimes works)
                    try:
                        # DeepExplainer needs the model's input layer
                        input_layer = self.model.layers[0]
                        
                        # Suppress Keras warnings during SHAP execution
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', category=UserWarning, module='keras')
                            warnings.filterwarnings('ignore', message='.*structure of `inputs`.*')
                            warnings.filterwarnings('ignore', message='.*The structure.*')
                            explainer = shap.DeepExplainer(self.model, background.astype(np.float32))
                            shap_values = explainer.shap_values(images_to_explain.astype(np.float32))
                        
                        if isinstance(shap_values, list):
                            preds = self.model.predict(images_to_explain, verbose=0)
                            pred_classes = np.argmax(preds, axis=1)
                            shap_values_selected = np.array([shap_values[cls][i] for i, cls in enumerate(pred_classes)])
                        else:
                            shap_values_selected = shap_values
                        
                        explainer_used = "DeepExplainer"
                        logger.info("DeepExplainer succeeded")
                        
                    except Exception as e3:
                        logger.error(f"All SHAP explainers failed. Last error: {e3}")
                        raise ValueError(f"Could not generate SHAP explanations. Errors: GradientExplainer={e}, PartitionExplainer={e2}, DeepExplainer={e3}")
            
            if shap_values_selected is None:
                raise ValueError("Failed to generate SHAP values with any explainer")
            
            # Generate visualizations
            shap_plots = []
            preds = self.model.predict(images_to_explain, verbose=0)
            
            for i, idx in enumerate(explain_indices):
                try:
                    # Create SHAP plot for this image
                    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                    
                    # Original image
                    axes[0].imshow(images_to_explain[i])
                    axes[0].set_title(f'Original Image {i+1}')
                    axes[0].axis('off')
                    
                    # SHAP values (for multi-class, show explanation for predicted class)
                    pred_class = np.argmax(preds[i])
                    
                    # Extract SHAP values for this image and predicted class
                    if isinstance(shap_values_selected, np.ndarray):
                        if len(shap_values_selected.shape) == 5:
                            # Shape: (batch, height, width, channels, classes)
                            shap_img = shap_values_selected[i, :, :, :, pred_class]
                        elif len(shap_values_selected.shape) == 4:
                            # Shape: (batch, height, width, channels) - single class or already selected
                            shap_img = shap_values_selected[i]
                        else:
                            # Try to reshape or use sum across channels
                            shap_img = shap_values_selected[i] if len(shap_values_selected.shape) >= 3 else shap_values_selected
                        
                        # If multi-channel, take mean or sum across channels for visualization
                        if len(shap_img.shape) == 3 and shap_img.shape[2] > 1:
                            shap_img = np.sum(np.abs(shap_img), axis=2)  # Sum absolute values across channels
                        
                        # Normalize for visualization
                        shap_img = (shap_img - shap_img.min()) / (shap_img.max() - shap_img.min() + 1e-8)
                        
                        im = axes[1].imshow(shap_img, cmap='hot', alpha=0.8)
                        axes[1].imshow(images_to_explain[i], alpha=0.3)
                        axes[1].set_title(f'SHAP Explanation (Class: {self.class_names[pred_class]})')
                        axes[1].axis('off')
                        plt.colorbar(im, ax=axes[1], fraction=0.046)
                    else:
                        # Fallback: use SHAP's built-in image plot
                        try:
                            shap.image_plot(shap_values_selected[i:i+1], images_to_explain[i:i+1], show=False, ax=axes[1])
                        except:
                            axes[1].text(0.5, 0.5, 'SHAP visualization\nnot available', 
                                        ha='center', va='center', transform=axes[1].transAxes)
                    
                    plt.tight_layout()
                    
                    # Save plot
                    plot_path = self.output_dir / f"shap_explanation_{i+1}.png"
                    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    shap_plots.append(str(plot_path))
                    
                except Exception as e:
                    logger.warning(f"Failed to create SHAP plot for image {i}: {e}")
                    continue
            
            results = {
                'method': 'SHAP',
                'explainer_used': explainer_used,
                'num_explained': num_explain,
                'background_samples': background_size,
                'plots': shap_plots,
                'success': True
            }
            
            logger.info(f"SHAP explanations generated successfully for {num_explain} images")
            return results
            
        except Exception as e:
            logger.error(f"Error generating SHAP explanations: {e}", exc_info=True)
            return {'error': str(e), 'success': False}
    
    def generate_lime_explanations(
        self,
        images: np.ndarray,
        file_paths: List[str],
        true_labels: np.ndarray,
        num_explanations: int = 5
    ) -> Dict[str, Any]:
        """
        Generate LIME explanations for images.
        
        Args:
            images: Array of images to explain
            file_paths: List of file paths for images
            true_labels: True labels for images
            num_explanations: Number of images to explain
            
        Returns:
            Dictionary containing LIME results
        """
        if not LIME_AVAILABLE:
            logger.warning("LIME not available. Skipping LIME explanations.")
            return {'error': 'LIME not available'}
        
        if len(images) == 0:
            logger.warning("No images provided for LIME explanations.")
            return {'error': 'No images provided'}
        
        try:
            logger.info("Generating LIME explanations...")
            
            # Create LIME explainer
            explainer = lime_image.LimeImageExplainer()
            
            # Wrapper function for model prediction
            def predict_fn(images_array):
                """Wrapper for model prediction compatible with LIME."""
                # LIME may pass images in different format
                if len(images_array.shape) == 4:
                    # Already batched
                    preds = self.model.predict(images_array, verbose=0)
                else:
                    # Single image, add batch dimension
                    preds = self.model.predict(np.expand_dims(images_array, 0), verbose=0)
                return preds
            
            num_explain = min(num_explanations, len(images))
            explain_indices = np.random.choice(len(images), num_explain, replace=False)
            
            lime_plots = []
            lime_results = []
            
            for i, idx in enumerate(explain_indices):
                try:
                    image = images[idx]
                    true_label = true_labels[idx] if idx < len(true_labels) else None
                    
                    # Get prediction
                    pred = self.model.predict(np.expand_dims(image, 0), verbose=0)
                    pred_class = np.argmax(pred[0])
                    pred_proba = pred[0][pred_class]
                    
                    # Explain
                    # Use fewer samples for faster processing (can be configured)
                    num_samples = 500  # Reduced from 1000 for faster processing
                    explanation = explainer.explain_instance(
                        image.astype(np.uint8),
                        predict_fn,
                        top_labels=len(self.class_names),
                        hide_color=0,
                        num_samples=num_samples
                    )
                    
                    # Get explanation for predicted class
                    temp, mask = explanation.get_image_and_mask(
                        pred_class,
                        positive_only=True,
                        num_features=10,
                        hide_rest=False
                    )
                    
                    # Create visualization
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
                    # Original image
                    axes[0].imshow(image)
                    title = f'Original Image {i+1}'
                    if true_label is not None:
                        title += f'\nTrue: {self.class_names[true_label]}'
                    axes[0].set_title(title)
                    axes[0].axis('off')
                    
                    # Prediction
                    axes[1].imshow(image)
                    axes[1].set_title(f'Predicted: {self.class_names[pred_class]}\nConfidence: {pred_proba:.2%}')
                    axes[1].axis('off')
                    
                    # LIME explanation
                    explained_img = mark_boundaries(temp, mask)
                    axes[2].imshow(explained_img)
                    axes[2].set_title('LIME Explanation\n(Green = Important)')
                    axes[2].axis('off')
                    
                    plt.tight_layout()
                    
                    # Save plot
                    plot_path = self.output_dir / f"lime_explanation_{i+1}.png"
                    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    lime_plots.append(str(plot_path))
                    
                    # Store explanation details
                    lime_results.append({
                        'image_index': int(idx),
                        'file_path': file_paths[idx] if idx < len(file_paths) else None,
                        'true_label': self.class_names[true_label] if true_label is not None else None,
                        'predicted_label': self.class_names[pred_class],
                        'confidence': float(pred_proba),
                        'plot_path': str(plot_path)
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to create LIME explanation for image {i}: {e}")
                    continue
            
            results = {
                'method': 'LIME',
                'num_explained': len(lime_plots),
                'plots': lime_plots,
                'explanations': lime_results,
                'success': True
            }
            
            logger.info(f"LIME explanations generated successfully for {len(lime_plots)} images")
            return results
            
        except Exception as e:
            logger.error(f"Error generating LIME explanations: {e}", exc_info=True)
            return {'error': str(e), 'success': False}
    
    def generate_interpretability_report(
        self,
        data_path: Path,
        split: str = 'test',
        max_samples: int = 10,
        shap_background_samples: int = 50,
        shap_max_evals: int = 100,
        lime_num_explanations: int = 5
    ) -> Dict[str, Any]:
        """
        Generate comprehensive interpretability report with SHAP and LIME.
        
        Args:
            data_path: Path to data directory
            split: Data split to analyze
            max_samples: Maximum number of sample images to load
            shap_background_samples: Number of background samples for SHAP
            shap_max_evals: Maximum evaluations for SHAP
            lime_num_explanations: Number of LIME explanations
            
        Returns:
            Dictionary containing interpretability results
        """
        logger.info(f"\n{'='*60}")
        logger.info("Running Model Interpretability Analysis...")
        logger.info(f"{'='*60}")
        
        # Load sample images
        images, file_paths, true_labels = self._load_sample_images(
            data_path, split, max_samples
        )
        
        if len(images) == 0:
            logger.error("No images loaded. Cannot perform interpretability analysis.")
            return {'error': 'No images loaded'}
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'split': split,
            'num_images_loaded': len(images),
            'class_names': self.class_names
        }
        
        # Generate SHAP explanations
        if SHAP_AVAILABLE:
            logger.info("Generating SHAP explanations...")
            shap_results = self.generate_shap_explanations(
                images,
                background_samples=shap_background_samples,
                max_evals=shap_max_evals
            )
            results['shap'] = shap_results
        else:
            results['shap'] = {'error': 'SHAP not available'}
        
        # Generate LIME explanations
        if LIME_AVAILABLE:
            logger.info("Generating LIME explanations...")
            lime_results = self.generate_lime_explanations(
                images,
                file_paths,
                true_labels,
                num_explanations=lime_num_explanations
            )
            results['lime'] = lime_results
        else:
            results['lime'] = {'error': 'LIME not available'}
        
        # Save results to JSON
        report_file = self.output_dir / "interpretability_report.json"
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Interpretability report saved to {report_file}")
        logger.info("Model interpretability analysis completed successfully.")
        
        return results

