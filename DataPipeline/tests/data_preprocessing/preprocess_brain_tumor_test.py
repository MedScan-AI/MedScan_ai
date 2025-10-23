"""
Test suite for Brain Tumor MRI preprocessing module.
Tests the BrainTumorPreprocessor class and related functionality.
"""

import pytest
from pathlib import Path
import tempfile
import shutil
import os
from PIL import Image
import numpy as np
import yaml

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../scripts/data_preprocessing'))

from process_braintumor import (
    ImageProcessor,
    BrainTumorPreprocessor,
    DatasetValidator
)


# Load configuration
def load_config():
    """Load configuration from vision_pipeline.yml"""
    config_path = Path(__file__).parent.parent.parent / "config" / "vision_pipeline.yml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


CONFIG = load_config()
PREPROCESSING_CONFIG = CONFIG['data_preprocessing']
COMMON_CONFIG = PREPROCESSING_CONFIG['common']
BRAIN_TUMOR_CONFIG = PREPROCESSING_CONFIG['brain_tumor_mri']


class TestImageProcessor:
    """Test cases for ImageProcessor base class."""
    
    @pytest.fixture
    def processor(self):
        """Fixture providing an ImageProcessor instance."""
        target_size = tuple(COMMON_CONFIG['target_size'])
        return ImageProcessor(target_size=target_size)
    
    @pytest.fixture
    def sample_image(self):
        """Fixture providing a sample test image."""
        # Create a simple test image
        img = Image.new('RGB', (512, 512), color=(73, 109, 137))
        return img
    
    def test_initialization(self, processor):
        """Test ImageProcessor initialization."""
        target_size = tuple(COMMON_CONFIG['target_size'])
        assert processor.target_size == target_size
        assert processor.logger is not None
    
    def test_resize_image(self, processor, sample_image):
        """Test image resizing functionality."""
        target_size = tuple(COMMON_CONFIG['target_size'])
        resized = processor.resize_image(sample_image)
        assert resized.size == target_size
        assert resized.mode == 'RGB'
    
    def test_normalize_image(self, processor, sample_image):
        """Test image normalization."""
        normalized = processor.normalize_image(sample_image)
        assert isinstance(normalized, np.ndarray)
        assert normalized.dtype == np.float32
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
    
    def test_denormalize_image(self, processor):
        """Test image denormalization."""
        # Create normalized array
        target_size = tuple(COMMON_CONFIG['target_size'])
        normalized = np.random.rand(target_size[1], target_size[0], 3).astype(np.float32)
        denormalized = processor.denormalize_image(normalized)
        assert isinstance(denormalized, Image.Image)
        assert denormalized.size == target_size


class TestBrainTumorPreprocessor:
    """Test cases for BrainTumorPreprocessor class."""
    
    @pytest.fixture
    def temp_dirs(self):
        """Fixture providing temporary directories for testing."""
        raw_dir = tempfile.mkdtemp()
        processed_dir = tempfile.mkdtemp()
        
        yield raw_dir, processed_dir
        
        # Cleanup
        shutil.rmtree(raw_dir, ignore_errors=True)
        shutil.rmtree(processed_dir, ignore_errors=True)
    
    @pytest.fixture
    def create_sample_dataset(self, temp_dirs):
        """Fixture creating a small sample dataset for testing."""
        raw_dir, _ = temp_dirs
        
        # Create directory structure from config
        classes = BRAIN_TUMOR_CONFIG['classes']
        splits = BRAIN_TUMOR_CONFIG['splits']
        sample_images_per_class = BRAIN_TUMOR_CONFIG['test']['sample_images_per_class']
        
        for split in splits:
            for class_name in classes:
                class_dir = Path(raw_dir) / split / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                
                # Create sample images per class (from config)
                for i in range(sample_images_per_class):
                    img = Image.new('RGB', (256, 256), color=(100 + i*20, 100, 100))
                    img.save(class_dir / f"sample_{i:03d}.jpg", 'JPEG')
        
        return raw_dir
    
    @pytest.fixture
    def preprocessor(self, temp_dirs, create_sample_dataset):
        """Fixture providing a BrainTumorPreprocessor instance."""
        raw_dir, processed_dir = temp_dirs
        target_size = tuple(COMMON_CONFIG['target_size'])
        quality = COMMON_CONFIG['quality']
        
        return BrainTumorPreprocessor(
            raw_data_path=raw_dir,
            preprocessed_data_path=processed_dir,
            target_size=target_size,
            quality=quality
        )
    
    def test_initialization(self, preprocessor):
        """Test BrainTumorPreprocessor initialization."""
        target_size = tuple(COMMON_CONFIG['target_size'])
        quality = COMMON_CONFIG['quality']
        classes = BRAIN_TUMOR_CONFIG['classes']
        splits = BRAIN_TUMOR_CONFIG['splits']
        
        assert preprocessor.target_size == target_size
        assert preprocessor.quality == quality
        assert preprocessor.classes == classes
        assert preprocessor.splits == splits
    
    def test_create_output_directories(self, preprocessor):
        """Test output directory creation."""
        preprocessor.create_output_directories()
        
        for split in preprocessor.splits:
            for class_name in preprocessor.classes:
                output_dir = preprocessor.preprocessed_data_path / split / class_name
                assert output_dir.exists()
                assert output_dir.is_dir()
    
    def test_get_image_files(self, preprocessor):
        """Test image file retrieval."""
        sample_images_per_class = BRAIN_TUMOR_CONFIG['test']['sample_images_per_class']
        image_files = preprocessor.get_image_files('Training', 'glioma')
        assert len(image_files) == sample_images_per_class
        assert all(f.suffix == '.jpg' for f in image_files)
    
    def test_preprocess_single_image(self, preprocessor):
        """Test preprocessing of a single image with UUID filename."""
        preprocessor.create_output_directories()
        target_size = tuple(COMMON_CONFIG['target_size'])
        
        image_files = preprocessor.get_image_files('Training', 'glioma')
        assert len(image_files) > 0
        
        input_path = image_files[0]
        # Generate UUID-based Patient ID for output filename
        patient_id = preprocessor.generate_patient_id()
        output_path = preprocessor.preprocessed_data_path / 'Training' / 'glioma' / f'{patient_id}.jpg'
        
        success = preprocessor.preprocess_image(input_path, output_path)
        assert success is True
        assert output_path.exists()
        
        # Verify output image
        output_img = Image.open(output_path)
        assert output_img.size == target_size
        
        # Verify Patient ID is UUID format (32 hex characters)
        assert len(patient_id) == 32
        assert all(c in '0123456789ABCDEF' for c in patient_id)
    
    def test_process_class(self, preprocessor):
        """Test processing all images in a class with UUID filenames."""
        preprocessor.create_output_directories()
        sample_images_per_class = BRAIN_TUMOR_CONFIG['test']['sample_images_per_class']
        
        processed_count = preprocessor.process_class('Training', 'glioma')
        assert processed_count == sample_images_per_class
        
        # Verify output files exist with UUID-based names
        output_dir = preprocessor.preprocessed_data_path / 'Training' / 'glioma'
        output_files = list(output_dir.glob('*.jpg'))
        assert len(output_files) == sample_images_per_class
        
        # Verify all filenames are UUID format (32 hex characters)
        for output_file in output_files:
            filename = output_file.stem
            assert len(filename) == 32
            assert all(c in '0123456789ABCDEF' for c in filename)
    
    def test_process_all(self, preprocessor):
        """Test processing entire dataset with UUID filenames."""
        stats = preprocessor.process_all()
        
        sample_images_per_class = BRAIN_TUMOR_CONFIG['test']['sample_images_per_class']
        num_classes = len(BRAIN_TUMOR_CONFIG['classes'])
        num_splits = len(BRAIN_TUMOR_CONFIG['splits'])
        expected_total = sample_images_per_class * num_classes * num_splits
        
        # Should process sample_images * classes * splits
        assert stats['total_processed'] == expected_total
        assert len(stats['by_class']) == num_classes * num_splits
        
        # Verify all output directories have files with UUID names
        for split in preprocessor.splits:
            for class_name in preprocessor.classes:
                output_dir = preprocessor.preprocessed_data_path / split / class_name
                output_files = list(output_dir.glob('*.jpg'))
                assert len(output_files) == sample_images_per_class
                
                # Verify UUID format for all files
                for output_file in output_files:
                    filename = output_file.stem
                    assert len(filename) == 32
                    assert all(c in '0123456789ABCDEF' for c in filename)


class TestBrainTumorPreprocessingRealData:
    """Test cases using actual raw data (if available)."""
    
    @pytest.fixture
    def real_data_paths(self):
        """Fixture providing paths to real data from config."""
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent
        
        raw_data_path = project_root / BRAIN_TUMOR_CONFIG['raw_data_path']
        test_output_path = project_root / BRAIN_TUMOR_CONFIG['test_output_path']
        
        return raw_data_path, test_output_path
    
    def test_preprocess_10_images_from_raw_dataset(self, real_data_paths):
        """
        Test preprocessing 10 random images from the actual raw dataset.
        This test picks 10 images from raw data and saves them to test directory.
        """
        raw_data_path, test_output_path = real_data_paths
        
        # Get config values
        splits = BRAIN_TUMOR_CONFIG['splits']
        classes = BRAIN_TUMOR_CONFIG['classes']
        num_test_images = BRAIN_TUMOR_CONFIG['test']['num_test_images']
        target_size = tuple(COMMON_CONFIG['target_size'])
        quality = COMMON_CONFIG['quality']
        
        # Check if raw data exists
        if not raw_data_path.exists():
            pytest.skip("Raw data not available for testing")
        
        # Find available images
        all_images = []
        for split in splits:
            for class_name in classes:
                class_dir = raw_data_path / split / class_name
                if class_dir.exists():
                    images = list(class_dir.glob('*.jpg'))[:3]  # Take up to 3 from each class
                    all_images.extend([(img, split, class_name) for img in images])
        
        # Select num_test_images from config (or fewer if not enough available)
        selected_images = all_images[:num_test_images]
        
        if len(selected_images) == 0:
            pytest.skip("No images found in raw dataset")
        
        # Create preprocessor for test
        preprocessor = BrainTumorPreprocessor(
            raw_data_path=str(raw_data_path),
            preprocessed_data_path=str(test_output_path),
            target_size=target_size,
            quality=quality
        )
        
        # Process selected images with UUID filenames
        processed_count = 0
        for img_path, split, class_name in selected_images:
            output_dir = test_output_path / split / class_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate UUID-based Patient ID for filename
            patient_id = preprocessor.generate_patient_id()
            output_path = output_dir / f"{patient_id}.jpg"
            success = preprocessor.preprocess_image(img_path, output_path)
            
            if success:
                processed_count += 1
                
                # Verify the preprocessed image
                assert output_path.exists()
                output_img = Image.open(output_path)
                assert output_img.size == target_size
                assert output_img.mode == 'RGB'
                
                # Verify Patient ID is UUID format
                assert len(patient_id) == 32
                assert all(c in '0123456789ABCDEF' for c in patient_id)
        
        assert processed_count >= 1, "Should process at least 1 image"
        print(f"\nSuccessfully preprocessed {processed_count} images from raw dataset")
        print(f"Output location: {test_output_path}")
        
        # Cleanup test output
        shutil.rmtree(test_output_path, ignore_errors=True)


class TestDatasetValidator:
    """Test cases for DatasetValidator class."""
    
    @pytest.fixture
    def temp_processed_dir(self):
        """Fixture providing a temporary processed directory."""
        temp_dir = tempfile.mkdtemp()
        
        # Create sample structure from config
        target_size = tuple(COMMON_CONFIG['target_size'])
        classes = BRAIN_TUMOR_CONFIG['classes'][:2]  # Use first 2 classes for testing
        splits = BRAIN_TUMOR_CONFIG['splits']
        
        for split in splits:
            for class_name in classes:
                class_dir = Path(temp_dir) / split / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                
                # Create sample images
                for i in range(2):
                    img = Image.new('RGB', target_size, color=(100, 100, 100))
                    img.save(class_dir / f"sample_{i}.jpg", 'JPEG')
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_validator_initialization(self, temp_processed_dir):
        """Test DatasetValidator initialization."""
        validator = DatasetValidator(temp_processed_dir)
        assert validator.preprocessed_data_path.exists()
    
    def test_validate_structure(self, temp_processed_dir):
        """Test structure validation."""
        validator = DatasetValidator(temp_processed_dir)
        splits = BRAIN_TUMOR_CONFIG['splits']
        classes = BRAIN_TUMOR_CONFIG['classes'][:2]  # First 2 classes used in fixture
        
        is_valid = validator.validate_structure(
            expected_splits=splits,
            expected_classes=classes
        )
        assert is_valid is True
    
    def test_validate_images(self, temp_processed_dir):
        """Test image validation."""
        validator = DatasetValidator(temp_processed_dir)
        target_size = tuple(COMMON_CONFIG['target_size'])
        is_valid = validator.validate_images(target_size=target_size)
        assert is_valid is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

