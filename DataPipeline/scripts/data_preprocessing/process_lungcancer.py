"""
Lung Cancer CT Scan Dataset Preprocessing Script

This module provides object-oriented classes for preprocessing lung cancer CT scan images.
It handles complex directory structures, class name normalization, and image standardization.
"""

import os
import re
import uuid
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Set
from datetime import datetime
from glob import glob
import logging
from PIL import Image
import numpy as np


class ImageProcessor:
    """Base class for image processing operations."""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize the ImageProcessor.
        
        Args:
            target_size: Target size for resizing images (width, height)
        """
        self.target_size = target_size
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_image(self, image_path: str) -> Optional[Image.Image]:
        """
        Load an image from the given path.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            PIL Image object or None if loading fails
        """
        try:
            img = Image.open(image_path)
            return img.convert('RGB')
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {str(e)}")
            return None
    
    def resize_image(self, image: Image.Image) -> Image.Image:
        """
        Resize image to target size.
        
        Args:
            image: PIL Image object
            
        Returns:
            Resized PIL Image object
        """
        return image.resize(self.target_size, Image.Resampling.LANCZOS)
    
    def normalize_image(self, image: Image.Image) -> np.ndarray:
        """
        Normalize image pixel values to [0, 1] range.
        
        Args:
            image: PIL Image object
            
        Returns:
            Normalized numpy array
        """
        img_array = np.array(image, dtype=np.float32)
        return img_array / 255.0
    
    def denormalize_image(self, img_array: np.ndarray) -> Image.Image:
        """
        Convert normalized array back to PIL Image.
        
        Args:
            img_array: Normalized numpy array
            
        Returns:
            PIL Image object
        """
        img_array = np.clip(img_array * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)
    
    def process_single_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Process a single image through the complete pipeline.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Normalized numpy array or None if processing fails
        """
        image = self.load_image(image_path)
        if image is None:
            return None
        
        resized = self.resize_image(image)
        normalized = self.normalize_image(resized)
        
        return normalized


class ClassNameNormalizer:
    """Utility class for normalizing lung cancer class names."""
    
    # Mapping from various input class names to standardized output names
    CLASS_MAPPING = {
        'adenocarcinoma': 'adenocarcinoma',
        'bengin cases': 'benign',
        'bengincases': 'benign',
        'bengin case': 'benign',
        'large.cell.carcinoma': 'large_cell_carcinoma',
        'malignant cases': 'malignant',
        'malignantcases': 'malignant',
        'malignant case': 'malignant',
        'normal': 'normal',
        'squamous.cell.carcinoma': 'squamous_cell_carcinoma',
    }
    
    @classmethod
    def normalize(cls, class_name: str) -> str:
        """
        Normalize a class name to standard format.
        
        Args:
            class_name: Original class name
            
        Returns:
            Normalized class name
        """
        # Convert to lowercase and remove extra spaces
        normalized = class_name.lower().strip()
        
        # Handle long class names with metadata (e.g., "adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib")
        # Extract just the cancer type
        if '_' in normalized and any(base in normalized for base in ['adenocarcinoma', 'carcinoma']):
            normalized = normalized.split('_')[0]
        
        # Apply mapping
        return cls.CLASS_MAPPING.get(normalized, normalized.replace(' ', '_').replace('.', '_'))


class LungCancerPreprocessor(ImageProcessor):
    """Preprocessor specifically for Lung Cancer CT Scan dataset."""
    
    def __init__(
        self,
        raw_data_path: str,
        preprocessed_data_path: str,
        target_size: Tuple[int, int] = (224, 224),
        quality: int = 95
    ):
        """
        Initialize the LungCancerPreprocessor.
        
        Args:
            raw_data_path: Path to raw lung cancer data
            preprocessed_data_path: Path to save preprocessed data
            target_size: Target size for resizing images
            quality: JPEG quality for saving images (1-100)
        """
        super().__init__(target_size)
        self.raw_data_path = Path(raw_data_path)
        self.preprocessed_data_path = Path(preprocessed_data_path)
        self.quality = quality
        self.class_normalizer = ClassNameNormalizer()
        self.discovered_classes: Set[str] = set()
        self.stats: Dict[str, int] = {
            'total_processed': 0,
            'total_failed': 0,
            'by_split_class': {}
        }
    
    def generate_patient_id(self) -> str:
        """
        Generate a unique Patient ID using UUID4.
        
        Returns:
            UUID-based Patient ID (32-character hexadecimal string)
        """
        # Generate UUID4 and convert to hex string (without dashes)
        patient_id = uuid.uuid4().hex.upper()
        return patient_id
    
    def discover_data_structure(self) -> Dict[str, List[Path]]:
        """
        Discover the actual data structure in the raw data directory.
        
        Returns:
            Dictionary mapping splits to list of class directories
        """
        structure = {}
        
        # Look for the main data directory
        data_dir = self.raw_data_path / "LungcancerDataSet" / "Data"
        
        if data_dir.exists():
            # Standard structure: Data/train, Data/test, Data/valid
            for split_dir in data_dir.iterdir():
                if split_dir.is_dir():
                    split_name = split_dir.name
                    structure[split_name] = []
                    
                    # Get all class directories in this split
                    for class_dir in split_dir.iterdir():
                        if class_dir.is_dir():
                            structure[split_name].append(class_dir)
                            normalized_class = self.class_normalizer.normalize(class_dir.name)
                            self.discovered_classes.add(normalized_class)
                            self.logger.info(
                                f"Discovered: {split_name}/{class_dir.name} -> {normalized_class}"
                            )
        
        # Also check for "Test cases" directory at the root level
        test_cases_dir = self.raw_data_path / "LungcancerDataSet" / "Test cases"
        if test_cases_dir.exists() and test_cases_dir.is_dir():
            structure['test_cases'] = [test_cases_dir]
            self.logger.info(f"Discovered: test_cases directory")
        
        return structure
    
    def create_output_directories(self, structure: Dict[str, List[Path]]) -> None:
        """
        Create necessary output directory structure.
        
        Args:
            structure: Dictionary with discovered data structure
        """
        for split, class_dirs in structure.items():
            for class_dir in class_dirs:
                if split == 'test_cases':
                    # Special handling for test_cases
                    output_dir = self.preprocessed_data_path / split / 'unlabeled'
                else:
                    normalized_class = self.class_normalizer.normalize(class_dir.name)
                    output_dir = self.preprocessed_data_path / split / normalized_class
                
                output_dir.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Created directory: {output_dir}")
    
    def get_image_files(self, directory: Path) -> List[Path]:
        """
        Get all image files from a directory.
        
        Args:
            directory: Directory to search
            
        Returns:
            List of image file paths (deduplicated)
        """
        if not directory.exists():
            self.logger.warning(f"Directory not found: {directory}")
            return []
        
        # Support multiple image formats
        # Use lowercase patterns only and deduplicate to handle case-insensitive filesystems
        image_extensions = ['*.jpg', '*.jpeg', '*.png']
        image_files = set()  # Use set to avoid duplicates
        
        for ext in image_extensions:
            image_files.update(directory.glob(ext))
            # Also check uppercase on case-sensitive systems
            image_files.update(directory.glob(ext.upper()))
        
        return sorted(list(image_files))
    
    def preprocess_image(
        self,
        image_path: Path,
        output_path: Path
    ) -> bool:
        """
        Preprocess a single image and save to output path.
        
        Args:
            image_path: Path to input image
            output_path: Path to save preprocessed image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load and process image
            image = self.load_image(str(image_path))
            if image is None:
                return False
            
            # Resize image
            resized = self.resize_image(image)
            
            # Save preprocessed image as JPEG (standardize format)
            # Change extension to .jpg for consistency
            output_path = output_path.with_suffix('.jpg')
            resized.save(output_path, 'JPEG', quality=self.quality, optimize=True)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error preprocessing {image_path}: {str(e)}")
            return False
    
    def process_class_directory(
        self,
        split: str,
        class_dir: Path,
        output_class_name: str
    ) -> int:
        """
        Process all images in a class directory.
        Uses UUID-based Patient IDs as filenames.
        
        Args:
            split: Data split name (train/test/valid)
            class_dir: Path to class directory
            output_class_name: Normalized output class name
            
        Returns:
            Number of successfully processed images
        """
        self.logger.info(f"Processing {split}/{class_dir.name}...")
        
        image_files = self.get_image_files(class_dir)
        if not image_files:
            self.logger.warning(f"No images found in {class_dir}")
            return 0
        
        successful = 0
        failed = 0
        
        for image_path in image_files:
            # Generate unique UUID-based Patient ID as filename
            patient_id = self.generate_patient_id()
            output_path = (
                self.preprocessed_data_path / split / output_class_name / f"{patient_id}.jpg"
            )
            
            if self.preprocess_image(image_path, output_path):
                successful += 1
            else:
                failed += 1
        
        self.logger.info(
            f"Completed {split}/{class_dir.name}: "
            f"{successful} successful, {failed} failed"
        )
        
        return successful
    
    def process_all(self) -> Dict[str, int]:
        """
        Process all images in the dataset.
        
        Returns:
            Statistics dictionary with processing results
        """
        self.logger.info("Starting Lung Cancer CT Scan preprocessing...")
        self.logger.info(f"Raw data path: {self.raw_data_path}")
        self.logger.info(f"Preprocessed data path: {self.preprocessed_data_path}")
        
        # Discover data structure
        structure = self.discover_data_structure()
        
        if not structure:
            self.logger.error("No data structure found!")
            return self.stats
        
        self.logger.info(f"\nDiscovered classes: {sorted(self.discovered_classes)}")
        
        # Create output directories
        self.create_output_directories(structure)
        
        # Process each split and class
        total_processed = 0
        
        for split, class_dirs in structure.items():
            for class_dir in class_dirs:
                if split == 'test_cases':
                    output_class_name = 'unlabeled'
                else:
                    output_class_name = self.class_normalizer.normalize(class_dir.name)
                
                processed = self.process_class_directory(split, class_dir, output_class_name)
                total_processed += processed
                
                # Update stats
                key = f"{split}_{output_class_name}"
                if key in self.stats['by_split_class']:
                    self.stats['by_split_class'][key] += processed
                else:
                    self.stats['by_split_class'][key] = processed
        
        self.stats['total_processed'] = total_processed
        
        self.logger.info(f"\nPreprocessing complete!")
        self.logger.info(f"Total images processed: {total_processed}")
        
        return self.stats
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get preprocessing statistics.
        
        Returns:
            Dictionary with statistics
        """
        return self.stats


class DatasetValidator:
    """Validator for checking preprocessed dataset integrity."""
    
    def __init__(self, preprocessed_data_path: str):
        """
        Initialize the DatasetValidator.
        
        Args:
            preprocessed_data_path: Path to preprocessed data
        """
        self.preprocessed_data_path = Path(preprocessed_data_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def validate_structure(self) -> bool:
        """
        Validate directory structure.
        
        Returns:
            True if structure is valid, False otherwise
        """
        self.logger.info("Validating directory structure...")
        
        if not self.preprocessed_data_path.exists():
            self.logger.error(f"Preprocessed path does not exist: {self.preprocessed_data_path}")
            return False
        
        valid = True
        
        # Check all splits and classes
        for split_dir in self.preprocessed_data_path.iterdir():
            if split_dir.is_dir():
                self.logger.info(f"\nSplit: {split_dir.name}")
                
                for class_dir in split_dir.iterdir():
                    if class_dir.is_dir():
                        count = len(list(class_dir.glob('*.jpg')))
                        self.logger.info(f"  {class_dir.name}: {count} images")
                        
                        if count == 0:
                            self.logger.warning(f"  No images in {class_dir}")
        
        return valid
    
    def validate_images(self, target_size: Tuple[int, int]) -> bool:
        """
        Validate that all images are properly formatted.
        
        Args:
            target_size: Expected image size
            
        Returns:
            True if all images are valid, False otherwise
        """
        self.logger.info("\nValidating image formats and sizes...")
        
        valid = True
        checked = 0
        errors = 0
        
        for image_path in self.preprocessed_data_path.rglob('*.jpg'):
            try:
                img = Image.open(image_path)
                if img.size != target_size:
                    self.logger.error(
                        f"Invalid size for {image_path}: {img.size} != {target_size}"
                    )
                    valid = False
                    errors += 1
                checked += 1
            except Exception as e:
                self.logger.error(f"Error validating {image_path}: {str(e)}")
                valid = False
                errors += 1
        
        self.logger.info(f"Validated {checked} images ({errors} errors)")
        
        return valid
    
    def get_class_distribution(self) -> Dict[str, Dict[str, int]]:
        """
        Get distribution of images across classes and splits.
        
        Returns:
            Dictionary with class distribution
        """
        distribution = {}
        
        for split_dir in self.preprocessed_data_path.iterdir():
            if split_dir.is_dir():
                split_name = split_dir.name
                distribution[split_name] = {}
                
                for class_dir in split_dir.iterdir():
                    if class_dir.is_dir():
                        count = len(list(class_dir.glob('*.jpg')))
                        distribution[split_name][class_dir.name] = count
        
        return distribution


def discover_partitions(base_path: Path) -> List[Dict[str, any]]:
    """
    Discover all year/month/day partitions in a base path.
    
    Args:
        base_path: Base directory to search
        
    Returns:
        List of partition info dicts with path and timestamp
    """
    partitions = []
    
    if not base_path.exists():
        return partitions
    
    # Find all YYYY/MM/DD directories
    pattern = str(base_path / "[0-9][0-9][0-9][0-9]" / "[0-9][0-9]" / "[0-9][0-9]")
    partition_dirs = glob(pattern)
    
    for partition_dir in sorted(partition_dirs):
        # Extract year/month/day from path
        parts = Path(partition_dir).parts[-3:]
        try:
            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
            timestamp = datetime(year, month, day)
            
            partitions.append({
                'path': Path(partition_dir),
                'timestamp': timestamp.isoformat(),
                'date_obj': timestamp
            })
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not parse partition path {partition_dir}: {e}")
            continue
    
    return partitions


def get_partition_path(base_path: Path, timestamp: datetime = None) -> Path:
    """
    Generate partition path based on year/month/day structure.
    
    Args:
        base_path: Base directory path
        timestamp: Optional timestamp (defaults to now)
        
    Returns:
        Full partition path: base_path/YYYY/MM/DD
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    partition_path = base_path / f"{timestamp.year:04d}" / f"{timestamp.month:02d}" / f"{timestamp.day:02d}"
    return partition_path


def main():
    """Main function to run the preprocessing pipeline with partition support."""
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(
        description='Preprocess lung cancer CT scan images with partition support'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/vision_pipeline.yml',
        help='Path to configuration file (default: config/vision_pipeline.yml)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    config_path = project_root / args.config
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get paths from config (with fallback to default)
    lung_cancer_config = None
    for dataset in config.get('data_acquisition', {}).get('kaggle', {}).get('datasets', []):
        if dataset.get('name') == 'lung_cancer_ct_scan':
            lung_cancer_config = dataset
            break
    
    if lung_cancer_config:
        raw_base_path = project_root / lung_cancer_config['download_path']
    else:
        raw_base_path = project_root / "data" / "raw" / "lung_cancer_ct_scan"
    
    preprocessed_base_path = project_root / "data" / "preprocessed" / "lung_cancer_ct_scan"
    
    # Discover raw partitions
    raw_partitions = discover_partitions(raw_base_path)
    
    if not raw_partitions:
        print(f"No partitioned data found in {raw_base_path}")
        print("Looking for data in base directory...")
        # Fallback to non-partitioned data
        if raw_base_path.exists() and any(raw_base_path.iterdir()):
            raw_data_path = raw_base_path
            preprocessed_data_path = preprocessed_base_path
            print(f"Using non-partitioned data from {raw_data_path}")
        else:
            print("No data found. Exiting.")
            return
    else:
        # Use the latest partition
        latest_partition = raw_partitions[-1]
        raw_data_path = latest_partition['path']
        
        # Write to same date partition in preprocessed
        preprocessed_data_path = get_partition_path(preprocessed_base_path, latest_partition['date_obj'])
        
        print(f"Found {len(raw_partitions)} partition(s)")
        print(f"Processing latest partition: {latest_partition['timestamp']}")
        print(f"  Raw data: {raw_data_path}")
        print(f"  Output to: {preprocessed_data_path}")
    
    # Initialize preprocessor
    preprocessor = LungCancerPreprocessor(
        raw_data_path=str(raw_data_path),
        preprocessed_data_path=str(preprocessed_data_path),
        target_size=(224, 224),
        quality=95
    )
    
    # Process all images
    stats = preprocessor.process_all()
    
    # Print statistics
    print("\n" + "="*50)
    print("PREPROCESSING STATISTICS")
    print("="*50)
    print(f"Total images processed: {stats['total_processed']}")
    print("\nBreakdown by split and class:")
    for key, count in sorted(stats['by_split_class'].items()):
        print(f"  {key}: {count}")
    print("="*50)
    
    # Validate preprocessed data
    validator = DatasetValidator(str(preprocessed_data_path))
    
    structure_valid = validator.validate_structure()
    
    if structure_valid:
        print("\n✓ Directory structure validation passed")
    else:
        print("\n✗ Directory structure validation failed")
    
    images_valid = validator.validate_images(target_size=(224, 224))
    
    if images_valid:
        print("✓ Image format validation passed")
    else:
        print("✗ Image format validation failed")
    
    # Print class distribution
    print("\n" + "="*50)
    print("CLASS DISTRIBUTION")
    print("="*50)
    distribution = validator.get_class_distribution()
    for split, classes in sorted(distribution.items()):
        print(f"\n{split}:")
        for class_name, count in sorted(classes.items()):
            print(f"  {class_name}: {count}")
    print("="*50)


if __name__ == "__main__":
    main()

