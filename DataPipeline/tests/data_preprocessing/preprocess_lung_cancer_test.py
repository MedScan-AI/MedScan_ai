"""
Test suite for Lung Cancer CT Scan preprocessing module.
Tests the LungCancerPreprocessor class and related functionality.
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

from process_lungcancer import (
    ImageProcessor,
    ClassNameNormalizer,
    LungCancerPreprocessor,
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
LUNG_CANCER_CONFIG = PREPROCESSING_CONFIG['lung_cancer_ct_scan']


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
        img = Image.new('RGB', (512, 512), color=(100, 150, 200))
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
        target_size = tuple(COMMON_CONFIG['target_size'])
        normalized = np.random.rand(target_size[1], target_size[0], 3).astype(np.float32)
        denormalized = processor.denormalize_image(normalized)
        assert isinstance(denormalized, Image.Image)
        assert denormalized.size == target_size


class TestClassNameNormalizer:
    """Test cases for ClassNameNormalizer class."""
    
    def test_normalize_adenocarcinoma(self):
        """Test normalization of adenocarcinoma class names."""
        assert ClassNameNormalizer.normalize('adenocarcinoma') == 'adenocarcinoma'
        assert ClassNameNormalizer.normalize('Adenocarcinoma') == 'adenocarcinoma'
        assert ClassNameNormalizer.normalize('adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib') == 'adenocarcinoma'
    
    def test_normalize_benign(self):
        """Test normalization of benign class names."""
        assert ClassNameNormalizer.normalize('Bengin cases') == 'benign'
        assert ClassNameNormalizer.normalize('BenginCases') == 'benign'
        assert ClassNameNormalizer.normalize('Bengin case') == 'benign'
    
    def test_normalize_malignant(self):
        """Test normalization of malignant class names."""
        assert ClassNameNormalizer.normalize('Malignant cases') == 'malignant'
        assert ClassNameNormalizer.normalize('MalignantCases') == 'malignant'
    
    def test_normalize_carcinoma(self):
        """Test normalization of carcinoma types."""
        assert ClassNameNormalizer.normalize('large.cell.carcinoma') == 'large_cell_carcinoma'
        assert ClassNameNormalizer.normalize('squamous.cell.carcinoma') == 'squamous_cell_carcinoma'
    
    def test_normalize_normal(self):
        """Test normalization of normal class."""
        assert ClassNameNormalizer.normalize('normal') == 'normal'
        assert ClassNameNormalizer.normalize('Normal') == 'normal'


class TestLungCancerPreprocessor:
    """Test cases for LungCancerPreprocessor class."""
    
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
        
        # Create directory structure mimicking lung cancer dataset
        base_dir = Path(raw_dir) / "LungcancerDataSet" / "Data"
        
        # Create train/test/valid splits with various class names
        splits_classes = {
            'train': ['adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib', 'Bengin cases', 'normal'],
            'test': ['adenocarcinoma', 'BenginCases', 'normal'],
            'valid': ['adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib', 'Bengin cases']
        }
        
        for split, classes in splits_classes.items():
            for class_name in classes:
                class_dir = base_dir / split / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                
                # Create 2 sample images (PNG and JPG)
                img1 = Image.new('RGB', (256, 256), color=(120, 120, 120))
                img1.save(class_dir / "sample_001.png", 'PNG')
                
                img2 = Image.new('RGB', (256, 256), color=(130, 130, 130))
                img2.save(class_dir / "sample_002.jpg", 'JPEG')
        
        # Create test cases directory
        test_cases_dir = Path(raw_dir) / "LungcancerDataSet" / "Test cases"
        test_cases_dir.mkdir(parents=True, exist_ok=True)
        img = Image.new('RGB', (256, 256), color=(140, 140, 140))
        img.save(test_cases_dir / "test_001.png", 'PNG')
        
        return raw_dir
    
    @pytest.fixture
    def preprocessor(self, temp_dirs, create_sample_dataset):
        """Fixture providing a LungCancerPreprocessor instance."""
        raw_dir, processed_dir = temp_dirs
        target_size = tuple(COMMON_CONFIG['target_size'])
        quality = COMMON_CONFIG['quality']
        
        return LungCancerPreprocessor(
            raw_data_path=raw_dir,
            preprocessed_data_path=processed_dir,
            target_size=target_size,
            quality=quality
        )
    
    def test_initialization(self, preprocessor):
        """Test LungCancerPreprocessor initialization."""
        target_size = tuple(COMMON_CONFIG['target_size'])
        quality = COMMON_CONFIG['quality']
        
        assert preprocessor.target_size == target_size
        assert preprocessor.quality == quality
        assert isinstance(preprocessor.class_normalizer, ClassNameNormalizer)
    
    def test_discover_data_structure(self, preprocessor):
        """Test data structure discovery."""
        structure = preprocessor.discover_data_structure()
        
        assert 'train' in structure
        assert 'test' in structure
        assert 'valid' in structure
        assert len(structure['train']) == 3
        assert len(structure['test']) == 3
        assert len(structure['valid']) == 2
    
    def test_create_output_directories(self, preprocessor):
        """Test output directory creation."""
        structure = preprocessor.discover_data_structure()
        preprocessor.create_output_directories(structure)
        
        # Verify directories were created
        assert (preprocessor.preprocessed_data_path / 'train' / 'adenocarcinoma').exists()
        assert (preprocessor.preprocessed_data_path / 'train' / 'benign').exists()
        assert (preprocessor.preprocessed_data_path / 'train' / 'normal').exists()
    
    def test_get_image_files(self, preprocessor):
        """Test image file retrieval."""
        structure = preprocessor.discover_data_structure()
        train_dirs = structure['train']
        
        for class_dir in train_dirs:
            image_files = preprocessor.get_image_files(class_dir)
            assert len(image_files) == 2  # 1 PNG + 1 JPG
    
    def test_preprocess_single_image(self, preprocessor):
        """Test preprocessing of a single image with UUID filename."""
        structure = preprocessor.discover_data_structure()
        preprocessor.create_output_directories(structure)
        target_size = tuple(COMMON_CONFIG['target_size'])
        
        # Get a sample image
        train_dirs = structure['train']
        image_files = preprocessor.get_image_files(train_dirs[0])
        assert len(image_files) > 0
        
        input_path = image_files[0]
        # Generate UUID-based Patient ID for output filename
        patient_id = preprocessor.generate_patient_id()
        output_path = preprocessor.preprocessed_data_path / f'{patient_id}.jpg'
        
        success = preprocessor.preprocess_image(input_path, output_path)
        assert success is True
        assert output_path.exists()
        
        # Verify output image
        output_img = Image.open(output_path)
        assert output_img.size == target_size
        output_img.close()  # Close image before cleanup
        
        # Verify Patient ID is UUID format (32 hex characters)
        assert len(patient_id) == 32
        assert all(c in '0123456789ABCDEF' for c in patient_id)
        
        # Cleanup
        output_path.unlink()
    
    def test_process_class_directory(self, preprocessor):
        """Test processing all images in a class directory with UUID filenames."""
        structure = preprocessor.discover_data_structure()
        preprocessor.create_output_directories(structure)
        
        train_dirs = structure['train']
        first_class_dir = train_dirs[0]
        output_class_name = preprocessor.class_normalizer.normalize(first_class_dir.name)
        
        processed_count = preprocessor.process_class_directory(
            'train',
            first_class_dir,
            output_class_name
        )
        
        assert processed_count == 2  # Should process 2 images
        
        # Verify output files exist with UUID-based names
        output_dir = preprocessor.preprocessed_data_path / 'train' / output_class_name
        output_files = list(output_dir.glob('*.jpg'))
        assert len(output_files) == 2
        
        # Verify all filenames are UUID format (32 hex characters)
        for output_file in output_files:
            filename = output_file.stem
            assert len(filename) == 32
            assert all(c in '0123456789ABCDEF' for c in filename)
    
    def test_process_all(self, preprocessor):
        """Test processing entire dataset with UUID filenames."""
        stats = preprocessor.process_all()
        
        # Should process: 3 classes * 2 images in train + 3 * 2 in test + 2 * 2 in valid + 1 in test_cases
        # = 6 + 6 + 4 + 1 = 17 images
        assert stats['total_processed'] == 17
        assert len(stats['by_split_class']) > 0
        
        # Verify all output files have UUID format names
        for split_dir in preprocessor.preprocessed_data_path.iterdir():
            if split_dir.is_dir():
                for class_dir in split_dir.iterdir():
                    if class_dir.is_dir():
                        output_files = list(class_dir.glob('*.jpg'))
                        for output_file in output_files:
                            filename = output_file.stem
                            assert len(filename) == 32
                            assert all(c in '0123456789ABCDEF' for c in filename)


class TestLungCancerPreprocessingRealData:
    """Test cases using actual raw data (if available)."""
    
    @pytest.fixture
    def real_data_paths(self):
        """Fixture providing paths to real data from config."""
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent
        
        raw_data_path = project_root / LUNG_CANCER_CONFIG['raw_data_path']
        test_output_path = project_root / LUNG_CANCER_CONFIG['test_output_path']
        
        return raw_data_path, test_output_path
    
    def test_preprocess_10_images_from_raw_dataset(self, real_data_paths):
        """
        Test preprocessing 10 random images from the actual raw dataset.
        This test picks 10 images from raw data and saves them to test directory.
        """
        raw_data_path, test_output_path = real_data_paths
        
        # Get config values
        expected_splits = LUNG_CANCER_CONFIG['expected_splits']
        num_test_images = LUNG_CANCER_CONFIG['test']['num_test_images']
        max_images_per_class = LUNG_CANCER_CONFIG['test']['max_images_per_class']
        target_size = tuple(COMMON_CONFIG['target_size'])
        quality = COMMON_CONFIG['quality']
        
        # Check if raw data exists
        if not raw_data_path.exists():
            pytest.skip("Raw data not available for testing")
        
        data_dir = raw_data_path / "LungcancerDataSet" / "Data"
        if not data_dir.exists():
            pytest.skip("LungcancerDataSet/Data directory not found")
        
        # Find available images from different splits and classes
        all_images = []
        
        for split_dir in data_dir.iterdir():
            if split_dir.is_dir() and split_dir.name in expected_splits:
                for class_dir in split_dir.iterdir():
                    if class_dir.is_dir():
                        # Get up to max_images_per_class from each class
                        images = []
                        images.extend(list(class_dir.glob('*.png'))[:max_images_per_class//2])
                        images.extend(list(class_dir.glob('*.jpg'))[:max_images_per_class//2])
                        
                        all_images.extend([
                            (img, split_dir.name, class_dir.name) 
                            for img in images
                        ])
        
        # Select num_test_images from config (or fewer if not enough available)
        selected_images = all_images[:num_test_images]
        
        if len(selected_images) == 0:
            pytest.skip("No images found in raw dataset")
        
        # Create preprocessor for test
        preprocessor = LungCancerPreprocessor(
            raw_data_path=str(raw_data_path),
            preprocessed_data_path=str(test_output_path),
            target_size=target_size,
            quality=quality
        )
        
        # Process selected images with UUID filenames
        processed_count = 0
        for img_path, split, class_name in selected_images:
            normalized_class = preprocessor.class_normalizer.normalize(class_name)
            output_dir = test_output_path / split / normalized_class
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
        splits = LUNG_CANCER_CONFIG['expected_splits'][:2]  # Use first 2 splits
        classes = LUNG_CANCER_CONFIG['expected_classes'][:3]  # Use first 3 classes
        
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
        is_valid = validator.validate_structure()
        assert is_valid is True
    
    def test_validate_images(self, temp_processed_dir):
        """Test image validation."""
        validator = DatasetValidator(temp_processed_dir)
        target_size = tuple(COMMON_CONFIG['target_size'])
        is_valid = validator.validate_images(target_size=target_size)
        assert is_valid is True
    
    def test_get_class_distribution(self, temp_processed_dir):
        """Test class distribution calculation."""
        validator = DatasetValidator(temp_processed_dir)
        distribution = validator.get_class_distribution()
        
        classes = LUNG_CANCER_CONFIG['expected_classes'][:3]
        splits = LUNG_CANCER_CONFIG['expected_splits'][:2]
        
        assert splits[0] in distribution  # 'train'
        assert splits[1] in distribution  # 'test'
        assert distribution[splits[0]][classes[0]] == 2  # 'adenocarcinoma'
        assert distribution[splits[0]][classes[1]] == 2  # 'benign'
        assert distribution[splits[1]][classes[2]] == 2  # 'normal'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

