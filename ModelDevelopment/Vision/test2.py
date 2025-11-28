"""
Demo script to generate mock SHAP and LIME visualizations for TB and Lung Cancer images.
Creates realistic-looking interpretability visualizations for demonstration purposes.
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image

# Configure logging first (before using logger)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import cv2 and skimage (optional)
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV (cv2) not available. Some smoothing features will be skipped.")

try:
    from skimage.segmentation import mark_boundaries
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    logger.warning("scikit-image not available. LIME boundary visualization will be simplified.")


def create_mock_shap_heatmap(image: np.ndarray, class_name: str, image_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Create a realistic-looking SHAP heatmap that highlights relevant regions.
    
    Args:
        image: Original image
        class_name: Class name (e.g., 'Tuberculosis', 'Normal', 'adenocarcinoma')
        image_size: Image size (height, width)
        
    Returns:
        SHAP heatmap array
    """
    height, width = image_size
    
    # Create base heatmap with Gaussian blobs in relevant regions
    heatmap = np.zeros((height, width), dtype=np.float32)
    
    # For TB images, highlight lung regions (center and upper portions)
    if 'Tuberculosis' in class_name or 'tb' in class_name.lower():
        # Create focused Gaussian blobs in specific lung regions (smaller, more distinct)
        # TB typically shows in upper and middle lung regions
        centers = [
            (width // 3, height // 4),      # Left lung upper (more focused)
            (2 * width // 3, height // 4),  # Right lung upper (more focused)
            (width // 3, height // 2),       # Left lung middle
            (2 * width // 3, height // 2),  # Right lung middle
        ]
        # Use smaller sigma for more focused highlights
        sigma = min(width, height) // 6  # Smaller sigma = more focused
        
        for idx, (cx, cy) in enumerate(centers):
            y, x = np.ogrid[:height, :width]
            dist_sq = (x - cx)**2 + (y - cy)**2
            # Vary intensity slightly for more realistic appearance
            intensity = 1.0 + (idx % 2) * 0.3  # Alternate between 1.0 and 1.3
            heatmap += intensity * np.exp(-dist_sq / (2 * sigma**2))
        
        # Add some lower-intensity background to create contrast
        # This ensures not everything is orange
        background = np.ones((height, width)) * 0.1
        heatmap = heatmap + background
        
        # Clip to prevent oversaturation
        heatmap = np.clip(heatmap, 0, 2.0)
    
    # For lung cancer, highlight suspicious nodules (random but realistic locations)
    elif any(cancer_type in class_name.lower() for cancer_type in ['adenocarcinoma', 'carcinoma', 'malignant', 'cancer']):
        # Create nodule-like highlights
        centers = [
            (width // 2, height // 3),
            (width // 3, height // 2),
            (2 * width // 3, 2 * height // 3),
        ]
        for cx, cy in centers:
            y, x = np.ogrid[:height, :width]
            dist_sq = (x - cx)**2 + (y - cy)**2
            sigma = min(width, height) // 6  # Smaller, more focused regions
            heatmap += 1.5 * np.exp(-dist_sq / (2 * sigma**2))
    
    # For normal/benign, create subtle, diffuse highlights
    else:
        # Subtle, uniform highlighting
        heatmap = np.ones((height, width)) * 0.3
        # Add some random variation
        noise = np.random.randn(height, width) * 0.1
        heatmap = np.clip(heatmap + noise, 0, 1)
    
    # Normalize to [0, 1] but preserve relative differences
    # Use percentile-based normalization for better contrast
    if heatmap.max() > 0:
        # For TB, use percentile to preserve hot spots
        if 'Tuberculosis' in class_name or 'tb' in class_name.lower():
            # Use 95th percentile to preserve the brightest regions
            p95 = np.percentile(heatmap, 95)
            if p95 > 0:
                heatmap = np.clip(heatmap / p95, 0, 1)
            else:
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-10)
        else:
            # Standard normalization for other classes
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-10)
    
    # Apply some smoothing for realism
    if CV2_AVAILABLE:
        heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
    else:
        # Simple smoothing using scipy if available
        try:
            from scipy.ndimage import gaussian_filter
            heatmap = gaussian_filter(heatmap, sigma=2)
        except ImportError:
            # Fallback: simple averaging (no external dependencies)
            kernel_size = 5
            h, w = heatmap.shape
            smoothed = np.zeros_like(heatmap)
            for i in range(kernel_size//2, h - kernel_size//2):
                for j in range(kernel_size//2, w - kernel_size//2):
                    smoothed[i, j] = np.mean(heatmap[i-kernel_size//2:i+kernel_size//2+1, 
                                                      j-kernel_size//2:j+kernel_size//2+1])
            heatmap = smoothed
    
    return heatmap


def create_mock_lime_explanation(image: np.ndarray, class_name: str, pred_class: str, 
                                 confidence: float, image_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Create a realistic-looking LIME explanation mask.
    
    Args:
        image: Original image
        class_name: True class name
        pred_class: Predicted class name
        confidence: Prediction confidence
        image_size: Image size (height, width)
        
    Returns:
        LIME mask array
    """
    height, width = image_size
    
    # Create segmentation mask with superpixel-like regions
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Create grid-based superpixels (simulating LIME's segmentation)
    grid_size = 20
    num_regions = (height // grid_size) * (width // grid_size)
    
    # Assign importance scores to regions
    if 'Tuberculosis' in class_name or 'Tuberculosis' in pred_class:
        # Highlight lung regions more strongly
        important_regions = np.random.choice(num_regions, size=num_regions // 3, replace=False)
    elif any(cancer_type in class_name.lower() or cancer_type in pred_class.lower() 
             for cancer_type in ['adenocarcinoma', 'carcinoma', 'malignant']):
        # Highlight nodule regions
        important_regions = np.random.choice(num_regions, size=num_regions // 4, replace=False)
    else:
        # Normal/benign - less focused highlights
        important_regions = np.random.choice(num_regions, size=num_regions // 2, replace=False)
    
    # Create mask based on important regions
    region_idx = 0
    for y in range(0, height, grid_size):
        for x in range(0, width, grid_size):
            if region_idx in important_regions:
                mask[y:y+grid_size, x:x+grid_size] = 255
            region_idx += 1
    
    # Apply some smoothing
    if CV2_AVAILABLE:
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
    else:
        # Simple smoothing using scipy if available
        try:
            from scipy.ndimage import gaussian_filter
            mask = gaussian_filter(mask.astype(float), sigma=1).astype(np.uint8)
        except ImportError:
            # Fallback: simple averaging
            try:
                # Simple 3x3 averaging filter
                kernel = np.ones((3, 3), np.float32) / 9
                mask_float = mask.astype(np.float32)
                # Manual convolution for 2D
                h, w = mask_float.shape
                smoothed = np.zeros_like(mask_float)
                for i in range(1, h-1):
                    for j in range(1, w-1):
                        smoothed[i, j] = np.mean(mask_float[i-1:i+2, j-1:j+2])
                mask = smoothed.astype(np.uint8)
            except:
                pass  # Skip smoothing if not available
    
    return mask


def load_sample_images(data_path: Path, dataset_name: str, split: str = 'test', max_samples: int = 5) -> List[Tuple[np.ndarray, str, str]]:
    """
    Load sample images from the data directory.
    
    Args:
        data_path: Path to data directory
        dataset_name: Name of dataset ('tb' or 'lung_cancer_ct_scan')
        split: Data split ('test', 'train', 'valid')
        max_samples: Maximum number of samples per class
        
    Returns:
        List of (image, class_name, file_path) tuples
    """
    images = []
    
    # Determine split path
    split_path = data_path / split
    logger.debug(f"Looking for split at: {split_path}")
    
    if not split_path.exists():
        # Try alternative paths
        if split == 'test':
            for alt in ['test', 'val', 'valid']:
                alt_path = data_path / alt
                if alt_path.exists():
                    split_path = alt_path
                    logger.debug(f"Found alternative split path: {split_path}")
                    break
    
    if not split_path.exists():
        logger.warning(f"Split path {split_path} does not exist")
        logger.warning(f"Available directories in {data_path}: {[d.name for d in data_path.iterdir() if d.is_dir()]}")
        return images
    
    # Load images from class directories
    class_dirs = sorted([d for d in split_path.iterdir() if d.is_dir()])
    
    for class_dir in class_dirs[:max_samples]:  # Limit classes
        class_name = class_dir.name
        image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpeg'))
        
        for img_file in image_files[:max_samples]:
            try:
                img = Image.open(img_file)
                img_array = np.array(img)
                
                # Resize if needed
                if img_array.shape[:2] != (224, 224):
                    img = img.resize((224, 224))
                    img_array = np.array(img)
                
                # Normalize to [0, 1] if needed
                if img_array.max() > 1.0:
                    img_array = img_array.astype(np.float32) / 255.0
                
                images.append((img_array, class_name, str(img_file)))
                
                if len(images) >= max_samples * len(class_dirs):
                    break
            except Exception as e:
                logger.warning(f"Failed to load {img_file}: {e}")
                continue
        
        if len(images) >= max_samples * len(class_dirs):
            break
    
    return images


def generate_mock_interpretability_reports(
    base_data_path: str,
    output_base_path: str,
    datasets: List[str] = ['tb', 'lung_cancer_ct_scan'],
    num_shap: int = 5,
    num_lime: int = 3
):
    """
    Generate mock SHAP and LIME visualizations for demo purposes.
    
    Args:
        base_data_path: Base path to preprocessed data
        output_base_path: Base path for output (ModelDevelopment/data)
        datasets: List of dataset names to process
        num_shap: Number of SHAP explanations to generate
        num_lime: Number of LIME explanations to generate
    """
    base_data_path = Path(base_data_path)
    output_base_path = Path(output_base_path)
    
    # Create output directory with timestamp
    now = datetime.now()
    year = now.strftime("%Y")
    month = now.strftime("%m")
    day = now.strftime("%d")
    timestamp = now.strftime("%H%M%S")
    
    for dataset_name in datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Generating mock interpretability reports for {dataset_name}")
        logger.info(f"{'='*60}")
        
        # Create output directory with dataset-specific subdirectory
        metadata_dir = output_base_path / year / month / day / timestamp
        interpretability_dir = metadata_dir / "interpretability_reports" / dataset_name
        interpretability_dir.mkdir(parents=True, exist_ok=True)
        
        # Load sample images
        data_path = base_data_path / dataset_name
        
        # If path doesn't exist, try alternative locations
        if not data_path.exists():
            # Try DataPipeline/data/preprocessed if running from project root
            # Get the script's directory to resolve relative paths
            script_dir = Path(__file__).parent.resolve()
            project_root = script_dir.parent.parent  # Go up from Vision/ to project root
            
            alt_paths = [
                project_root / "DataPipeline" / "data" / "preprocessed" / dataset_name,
                Path("DataPipeline/data/preprocessed") / dataset_name,
                Path("../DataPipeline/data/preprocessed") / dataset_name,
                Path("../../DataPipeline/data/preprocessed") / dataset_name,
            ]
            
            # Resolve all paths to absolute
            alt_paths = [p.resolve() if not p.is_absolute() else p for p in alt_paths]
            
            found = False
            for alt_path in alt_paths:
                if alt_path.exists():
                    data_path = alt_path
                    logger.info(f"Found data at alternative path: {data_path}")
                    found = True
                    break
            
            if not found:
                logger.error(f"Data path {base_data_path / dataset_name} does not exist. Skipping {dataset_name}.")
                logger.error(f"Tried paths:")
                logger.error(f"  - {base_data_path / dataset_name}")
                for alt_path in alt_paths:
                    logger.error(f"  - {alt_path}")
                logger.error(f"Please ensure the data path is correct. Expected structure:")
                logger.error(f"  {base_data_path / dataset_name}/YYYY/MM/DD/test/")
                logger.error(f"Or use: --data_path DataPipeline/data/preprocessed")
                continue
        
        # Find the latest partition directory (YYYY/MM/DD) - same logic as train_resnet.py
        latest_day = None
        try:
            # Find latest year
            years = [d for d in data_path.iterdir() if d.is_dir() and d.name.isdigit()]
            if not years:
                raise ValueError(f"No year directories found in {data_path}")
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
            
        except ValueError as e:
            logger.warning(f"Could not find data directory for {dataset_name}: {e}")
            continue
        except Exception as e:
            logger.warning(f"Error finding data directory for {dataset_name}: {e}")
            continue
        
        logger.info(f"Using data directory: {latest_day}")
        
        # Load images
        sample_images = load_sample_images(latest_day, dataset_name, split='test', max_samples=10)
        
        if len(sample_images) == 0:
            logger.warning(f"No images found for {dataset_name}. Skipping.")
            continue
        
        logger.info(f"Loaded {len(sample_images)} sample images")
        
        # Get class names
        class_names = sorted(list(set([img[1] for img in sample_images])))
        logger.info(f"Found classes: {class_names}")
        
        # Generate SHAP explanations
        shap_plots = []
        shap_images_to_explain = sample_images[:num_shap]
        
        for i, (image, class_name, file_path) in enumerate(shap_images_to_explain):
            try:
                # Create mock SHAP heatmap
                shap_heatmap = create_mock_shap_heatmap(image, class_name)
                
                # Create visualization
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                
                # Original image
                axes[0].imshow(image)
                axes[0].set_title(f'Original Image {i+1}\nClass: {class_name}')
                axes[0].axis('off')
                
                # SHAP explanation
                im = axes[1].imshow(shap_heatmap, cmap='hot', alpha=0.8, vmin=0, vmax=1)
                axes[1].imshow(image, alpha=0.3)
                axes[1].set_title(f'SHAP Explanation\n(Highlighted regions are important)')
                axes[1].axis('off')
                plt.colorbar(im, ax=axes[1], fraction=0.046)
                
                plt.tight_layout()
                
                # Save plot with dataset name prefix
                plot_path = interpretability_dir / f"{dataset_name}_shap_explanation_{i+1}.png"
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                shap_plots.append(str(plot_path))
                logger.info(f"Generated SHAP explanation {i+1}/{num_shap} for {dataset_name}")
                
            except Exception as e:
                logger.warning(f"Failed to create SHAP plot {i+1}: {e}")
                continue
        
        # Generate LIME explanations
        lime_plots = []
        lime_results = []
        lime_images_to_explain = sample_images[:num_lime]
        
        for i, (image, class_name, file_path) in enumerate(lime_images_to_explain):
            try:
                # Simulate prediction (for demo, use class_name as prediction)
                pred_class = class_name
                confidence = np.random.uniform(0.75, 0.95)  # Realistic confidence
                
                # Create mock LIME mask
                lime_mask = create_mock_lime_explanation(image, class_name, pred_class, confidence)
                
                # Create visualization
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Original image
                axes[0].imshow(image)
                axes[0].set_title(f'Original Image {i+1}\nTrue: {class_name}')
                axes[0].axis('off')
                
                # Prediction
                axes[1].imshow(image)
                axes[1].set_title(f'Predicted: {pred_class}\nConfidence: {confidence:.2%}')
                axes[1].axis('off')
                
                # LIME explanation (overlay mask on image)
                if SKIMAGE_AVAILABLE:
                    try:
                        # Convert mask to boundaries
                        explained_img = mark_boundaries(image, lime_mask > 127, color=(0, 1, 0), mode='thick')
                        axes[2].imshow(explained_img)
                    except:
                        # Fallback
                        axes[2].imshow(image)
                        axes[2].imshow(lime_mask, alpha=0.4, cmap='Greens')
                else:
                    # Fallback if skimage not available
                    axes[2].imshow(image)
                    axes[2].imshow(lime_mask, alpha=0.4, cmap='Greens')
                
                axes[2].set_title('LIME Explanation\n(Green = Important Regions)')
                axes[2].axis('off')
                
                plt.tight_layout()
                
                # Save plot with dataset name prefix
                plot_path = interpretability_dir / f"{dataset_name}_lime_explanation_{i+1}.png"
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                lime_plots.append(str(plot_path))
                lime_results.append({
                    'image_index': i,
                    'file_path': file_path,
                    'true_label': class_name,
                    'predicted_label': pred_class,
                    'confidence': float(confidence),
                    'plot_path': str(plot_path)
                })
                
                logger.info(f"Generated LIME explanation {i+1}/{num_lime} for {dataset_name}")
                
            except Exception as e:
                logger.warning(f"Failed to create LIME plot {i+1}: {e}")
                continue
        
        # Create interpretability report JSON
        report = {
            'timestamp': datetime.now().isoformat(),
            'split': 'test',
            'num_images_loaded': len(sample_images),
            'class_names': class_names,
            'shap': {
                'method': 'SHAP',
                'num_explained': len(shap_plots),
                'background_samples': 20,
                'plots': shap_plots,
                'success': True
            },
            'lime': {
                'method': 'LIME',
                'num_explained': len(lime_plots),
                'plots': lime_plots,
                'explanations': lime_results,
                'success': True
            }
        }
        
        # Save report with dataset name
        report_file = interpretability_dir / f"{dataset_name}_interpretability_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Interpretability report saved to {report_file}")
        logger.info(f"Generated {len(shap_plots)} SHAP and {len(lime_plots)} LIME explanations for {dataset_name}")
        logger.info(f"All files saved to: {interpretability_dir}")
    
    logger.info(f"\n{'='*60}")
    logger.info("Mock interpretability reports generation completed!")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate mock SHAP and LIME visualizations for demo')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Base path to preprocessed data (e.g., DataPipeline/data/preprocessed)')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Base path for output (e.g., ModelDevelopment/data)')
    parser.add_argument('--datasets', nargs='+', default=['tb', 'lung_cancer_ct_scan'],
                        help='Datasets to process')
    parser.add_argument('--num_shap', type=int, default=5,
                        help='Number of SHAP explanations to generate')
    parser.add_argument('--num_lime', type=int, default=3,
                        help='Number of LIME explanations to generate')
    
    args = parser.parse_args()
    
    generate_mock_interpretability_reports(
        base_data_path=args.data_path,
        output_base_path=args.output_path,
        datasets=args.datasets,
        num_shap=args.num_shap,
        num_lime=args.num_lime
    )

