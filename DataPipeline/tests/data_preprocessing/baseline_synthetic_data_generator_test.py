"""
Test Cases for Baseline Synthetic Patient Data Generator

This module tests the synthetic data generation for medical images,
including Patient ID extraction, data generation, and CSV output.
"""

import os
import sys
import pytest
import uuid
import csv
import shutil
import tempfile
import yaml
from pathlib import Path
from PIL import Image
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.data_preprocessing.baseline_synthetic_data_generator import (
    PatientDataGenerator,
    TBDataGenerator,
    LungCancerDataGenerator,
    SyntheticDataPipeline
)


class TestPatientDataGenerator:
    """Test cases for PatientDataGenerator base class."""
    
    @pytest.fixture
    def config(self):
        """Fixture providing test configuration."""
        return {
            'output': {
                'format': 'csv',
                'random_seed': 42,
                'include_image_path': True
            },
            'faker': {
                'locale': 'en_US'
            },
            'demographics': {
                'age_range': {'min': 22, 'max': 85},
                'weight_range': {'min': 45, 'max': 120},
                'height_range': {'min': 150, 'max': 195},
                'gender_distribution': {'Male': 0.48, 'Female': 0.48, 'Other': 0.04}
            },
            'general': {
                'max_symptoms_per_patient': 3,
                'max_medications_per_patient': 3,
                'max_surgeries_per_patient': 2,
                'prob_no_medication': 0.2,
                'prob_no_surgery': 0.5
            },
            'tb': {
                'examination_type': 'X-ray',
                'body_region': 'Chest',
                'presenting_symptoms': {
                    'Tuberculosis': ['Persistent cough', 'Seizures', 'Memory problems']
                },
                'current_medications': ['Dexamethasone (steroid)', 'None'],
                'previous_surgeries': ['None', 'Craniotomy'],
                'urgency_levels': {'Routine': 0.6, 'Urgent': 0.3, 'Emergent': 0.1}
            },
            'lung_cancer': {
                'examination_type': 'CT',
                'body_region': 'Chest',
                'presenting_symptoms': {
                    'adenocarcinoma': ['Persistent cough', 'Shortness of breath']
                },
                'current_medications': ['Carboplatin (chemotherapy)', 'None'],
                'previous_surgeries': ['None', 'Lobectomy'],
                'urgency_levels': {'Routine': 0.5, 'Urgent': 0.3, 'Emergent': 0.15, 'STAT': 0.05}
            }
        }
    
    @pytest.fixture
    def generator(self, config):
        """Fixture providing PatientDataGenerator instance."""
        return PatientDataGenerator(config)
    
    def test_initialization(self, generator, config):
        """Test generator initializes correctly."""
        assert generator.config == config
        assert generator.faker is not None
        assert generator.demographics == config['demographics']
        assert generator.general == config['general']
    
    def test_extract_patient_id_from_filename(self, generator):
        """Test Patient ID extraction from filename."""
        # Test UUID-style filename
        uuid_filename = "7F8B3C2E1D4A5F6E9C8B7A6D5E4F3C2B.jpg"
        patient_id = generator.extract_patient_id_from_filename(uuid_filename)
        assert patient_id == "7F8B3C2E1D4A5F6E9C8B7A6D5E4F3C2B"
        
        # Test with full path
        full_path = "data/test_preprocessed/tb/train/Normal/A1B2C3D4E5F6.jpg"
        patient_id = generator.extract_patient_id_from_filename(full_path)
        assert patient_id == "A1B2C3D4E5F6"
    
    def test_generate_patient_name(self, generator):
        """Test patient name generation using Faker."""
        name = generator.generate_patient_name()
        assert isinstance(name, str)
        assert len(name) > 0
        assert ' ' in name  # Should have first and last name
    
    def test_generate_age(self, generator):
        """Test age generation within configured range."""
        age = generator.generate_age()
        assert isinstance(age, int)
        assert 22 <= age <= 85
    
    def test_generate_weight(self, generator):
        """Test weight generation within configured range."""
        weight = generator.generate_weight()
        assert isinstance(weight, float)
        assert 45 <= weight <= 120
    
    def test_generate_height(self, generator):
        """Test height generation within configured range."""
        height = generator.generate_height()
        assert isinstance(height, int)
        assert 150 <= height <= 195
    
    def test_generate_gender(self, generator):
        """Test gender generation from distribution."""
        gender = generator.generate_gender()
        assert gender in ['Male', 'Female', 'Other']
    
    def test_generate_symptoms(self, generator):
        """Test symptom generation."""
        symptom_list = ['Headache', 'Nausea', 'Dizziness', 'Fatigue']
        symptoms = generator.generate_symptoms(symptom_list, max_symptoms=2)
        assert isinstance(symptoms, str)
        assert len(symptoms) > 0
    
    def test_generate_medications(self, generator):
        """Test medication generation."""
        med_list = ['Aspirin', 'Ibuprofen', 'None']
        medications = generator.generate_medications(med_list)
        assert isinstance(medications, str)
    
    def test_generate_surgeries(self, generator):
        """Test surgery generation."""
        surgery_list = ['Appendectomy', 'None']
        surgeries = generator.generate_surgeries(surgery_list)
        assert isinstance(surgeries, str)


class TestTBDataGenerator:
    """Test cases for TBDataGenerator."""
    
    @pytest.fixture
    def config(self):
        """Fixture providing test configuration."""
        return {
            'output': {
                'format': 'csv',
                'random_seed': 42,
                'include_image_path': True
            },
            'faker': {'locale': 'en_US'},
            'demographics': {
                'age_range': {'min': 22, 'max': 85},
                'weight_range': {'min': 45, 'max': 120},
                'height_range': {'min': 150, 'max': 195},
                'gender_distribution': {'Male': 0.48, 'Female': 0.48, 'Other': 0.04}
            },
            'general': {
                'max_symptoms_per_patient': 3,
                'max_medications_per_patient': 3,
                'max_surgeries_per_patient': 2,
                'prob_no_medication': 0.2,
                'prob_no_surgery': 0.5
            },
            'tb': {
                'examination_type': 'X-ray',
                'body_region': 'Chest',
                'presenting_symptoms': {
                    'Tuberculosis': ['Persistent cough', 'Coughing up blood', 'Night sweats'],
                    'Normal': ['Routine screening', 'None reported']
                },
                'current_medications': ['Isoniazid (TB treatment)', 'Rifampin', 'None'],
                'previous_surgeries': ['None', 'Lung biopsy'],
                'urgency_levels': {'Routine': 0.6, 'Urgent': 0.3, 'Emergent': 0.1}
            }
        }
    
    @pytest.fixture
    def generator(self, config):
        """Fixture providing TBDataGenerator instance."""
        return TBDataGenerator(config)
    
    def test_initialization(self, generator, config):
        """Test TB generator initializes correctly."""
        assert generator.tb_config == config['tb']
    
    def test_generate_patient_record(self, generator):
        """Test generating a complete patient record."""
        image_path = "data/test_preprocessed/tb/train/Normal/7F8B3C2E1D4A5F6E9C8B7A6D5E4F3C2B.jpg"
        class_name = "Normal"
        
        record = generator.generate_patient_record(image_path, class_name)
        
        # Check all required fields
        assert 'Patient_Full_Name' in record
        assert 'Patient_ID' in record
        assert 'Presenting_Symptoms' in record
        assert 'Current_Medications' in record
        assert 'Previous_Relevant_Surgeries' in record
        assert 'Age_Years' in record
        assert 'Weight_KG' in record
        assert 'Height_CM' in record
        assert 'Gender' in record
        assert 'Examination_Type' in record
        assert 'Body_Region' in record
        assert 'Urgency_Level' in record
        assert 'Image_Path' in record
        assert 'Diagnosis_Class' in record
        
        # Check Patient ID matches filename
        assert record['Patient_ID'] == '7F8B3C2E1D4A5F6E9C8B7A6D5E4F3C2B'
        
        # Check specific values
        assert record['Examination_Type'] == 'X-ray'
        assert record['Body_Region'] == 'Chest'
        assert record['Diagnosis_Class'] == 'Normal'
        assert record['Image_Path'] == image_path
        
        # Check data types
        assert isinstance(record['Age_Years'], int)
        assert isinstance(record['Weight_KG'], float)
        assert isinstance(record['Height_CM'], int)
        assert record['Gender'] in ['Male', 'Female', 'Other']
        assert record['Urgency_Level'] in ['Routine', 'Urgent', 'Emergent']


class TestLungCancerDataGenerator:
    """Test cases for LungCancerDataGenerator."""
    
    @pytest.fixture
    def config(self):
        """Fixture providing test configuration."""
        return {
            'output': {
                'format': 'csv',
                'random_seed': 42,
                'include_image_path': True
            },
            'faker': {'locale': 'en_US'},
            'demographics': {
                'age_range': {'min': 22, 'max': 85},
                'weight_range': {'min': 45, 'max': 120},
                'height_range': {'min': 150, 'max': 195},
                'gender_distribution': {'Male': 0.48, 'Female': 0.48, 'Other': 0.04}
            },
            'general': {
                'max_symptoms_per_patient': 3,
                'max_medications_per_patient': 3,
                'max_surgeries_per_patient': 2,
                'prob_no_medication': 0.2,
                'prob_no_surgery': 0.5
            },
            'lung_cancer': {
                'examination_type': 'CT',
                'body_region': 'Chest',
                'presenting_symptoms': {
                    'adenocarcinoma': ['Persistent cough', 'Shortness of breath'],
                    'normal': ['Routine screening', 'None reported']
                },
                'current_medications': ['Carboplatin (chemotherapy)', 'None'],
                'previous_surgeries': ['None', 'Lobectomy'],
                'urgency_levels': {'Routine': 0.5, 'Urgent': 0.3, 'Emergent': 0.15, 'STAT': 0.05}
            }
        }
    
    @pytest.fixture
    def generator(self, config):
        """Fixture providing LungCancerDataGenerator instance."""
        return LungCancerDataGenerator(config)
    
    def test_initialization(self, generator, config):
        """Test lung cancer generator initializes correctly."""
        assert generator.lung_config == config['lung_cancer']
    
    def test_generate_patient_record(self, generator):
        """Test generating a complete patient record."""
        image_path = "data/test_preprocessed/lung_cancer_ct_scan/train/adenocarcinoma/A1B2C3D4E5F67890A1B2C3D4E5F67890.jpg"
        class_name = "adenocarcinoma"
        
        record = generator.generate_patient_record(image_path, class_name)
        
        # Check all required fields
        assert 'Patient_Full_Name' in record
        assert 'Patient_ID' in record
        assert 'Presenting_Symptoms' in record
        assert 'Current_Medications' in record
        assert 'Previous_Relevant_Surgeries' in record
        assert 'Age_Years' in record
        assert 'Weight_KG' in record
        assert 'Height_CM' in record
        assert 'Gender' in record
        assert 'Examination_Type' in record
        assert 'Body_Region' in record
        assert 'Urgency_Level' in record
        assert 'Image_Path' in record
        assert 'Diagnosis_Class' in record
        
        # Check Patient ID matches filename
        assert record['Patient_ID'] == 'A1B2C3D4E5F67890A1B2C3D4E5F67890'
        
        # Check specific values
        assert record['Examination_Type'] == 'CT'
        assert record['Body_Region'] == 'Chest'
        assert record['Diagnosis_Class'] == 'adenocarcinoma'
        assert record['Image_Path'] == image_path
        
        # Check urgency level is valid
        assert record['Urgency_Level'] in ['Routine', 'Urgent', 'Emergent', 'STAT']


class TestSyntheticDataPipeline:
    """Test cases for SyntheticDataPipeline with test data."""
    
    @pytest.fixture(scope='class')
    def test_data_setup(self):
        """Set up test preprocessed data for both TB and lung cancer."""
        # Define test data paths
        test_base = Path('data/test_preprocessed')
        tb_base = test_base / 'tb'
        lung_base = test_base / 'lung_cancer_ct_scan'
        
        # TB test structure
        tb_classes = {
            'train': ['Normal', 'Tuberculosis'],
            'test': ['Normal', 'Tuberculosis']
        }
        
        # Lung cancer test structure
        lung_classes = {
            'train': ['adenocarcinoma', 'normal'],
            'test': ['benign']
        }
        
        # Create test images for TB
        tb_patient_ids = []
        for split, classes in tb_classes.items():
            for class_name in classes:
                output_dir = tb_base / split / class_name
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Create 3 test images per class
                for i in range(3):
                    patient_id = uuid.uuid4().hex.upper()
                    tb_patient_ids.append((patient_id, split, class_name))
                    
                    # Create a simple test image
                    img = Image.new('RGB', (224, 224), color=(100 + i*20, 100, 100))
                    img_path = output_dir / f"{patient_id}.jpg"
                    img.save(img_path, 'JPEG')
        
        # Create test images for lung cancer
        lung_patient_ids = []
        for split, classes in lung_classes.items():
            for class_name in classes:
                output_dir = lung_base / split / class_name
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Create 3 test images per class
                for i in range(3):
                    patient_id = uuid.uuid4().hex.upper()
                    lung_patient_ids.append((patient_id, split, class_name))
                    
                    # Create a simple test image
                    img = Image.new('RGB', (224, 224), color=(100, 100 + i*20, 100))
                    img_path = output_dir / f"{patient_id}.jpg"
                    img.save(img_path, 'JPEG')
        
        yield {
            'tb_base': tb_base,
            'lung_base': lung_base,
            'tb_patient_ids': tb_patient_ids,
            'lung_patient_ids': lung_patient_ids
        }
        
        # Cleanup after tests
        if test_base.exists():
            shutil.rmtree(test_base)
    
    def test_get_image_files_tb(self, test_data_setup):
        """Test getting image files for TB dataset."""
        from scripts.data_preprocessing.baseline_synthetic_data_generator import SyntheticDataPipeline
        
        # Create mock config in /tmp to avoid Mac file lock
        mock_config = {
            'output': {'format': 'csv', 'random_seed': 42, 'include_image_path': True},
            'faker': {'locale': 'en_US'},
            'demographics': {
                'age_range': {'min': 22, 'max': 85},
                'weight_range': {'min': 45, 'max': 120},
                'height_range': {'min': 150, 'max': 195},
                'gender_distribution': {'Male': 0.48, 'Female': 0.48, 'Other': 0.04}
            },
            'general': {
                'max_symptoms_per_patient': 3,
                'max_medications_per_patient': 3,
                'max_surgeries_per_patient': 2,
                'prob_no_medication': 0.2,
                'prob_no_surgery': 0.5
            },
            'tb': {
                'examination_type': 'X-ray',
                'body_region': 'Chest',
                'presenting_symptoms': {'Tuberculosis': ['Cough'], 'Normal': ['Screening']},
                'current_medications': ['None'],
                'previous_surgeries': ['None'],
                'urgency_levels': {'Routine': 1.0}
            },
            'lung_cancer': {
                'examination_type': 'CT',
                'body_region': 'Chest',
                'presenting_symptoms': {'adenocarcinoma': ['Cough'], 'normal': ['Screening']},
                'current_medications': ['None'],
                'previous_surgeries': ['None'],
                'urgency_levels': {'Routine': 1.0}
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump(mock_config, f)
            temp_config_path = f.name
        
        try:
            pipeline = SyntheticDataPipeline(temp_config_path)
            
            # Get test image files
            image_files = pipeline.get_image_files('data/test_preprocessed/tb')
            
            # Should have 12 images total (2 splits * 2 classes * 3 images)
            assert len(image_files) == 12
            
            # Check structure of returned data
            for img_path, split, class_name in image_files:
                assert isinstance(img_path, str)
                assert isinstance(split, str)
                assert isinstance(class_name, str)
                assert '.jpg' in img_path
        finally:
            os.unlink(temp_config_path)
    
    def test_get_image_files_lung(self, test_data_setup):
        """Test getting image files for lung cancer dataset."""
        from scripts.data_preprocessing.baseline_synthetic_data_generator import SyntheticDataPipeline
        
        # Create mock config in /tmp
        mock_config = {
            'output': {'format': 'csv', 'random_seed': 42, 'include_image_path': True},
            'faker': {'locale': 'en_US'},
            'demographics': {
                'age_range': {'min': 22, 'max': 85},
                'weight_range': {'min': 45, 'max': 120},
                'height_range': {'min': 150, 'max': 195},
                'gender_distribution': {'Male': 0.48, 'Female': 0.48, 'Other': 0.04}
            },
            'general': {
                'max_symptoms_per_patient': 3,
                'max_medications_per_patient': 3,
                'max_surgeries_per_patient': 2,
                'prob_no_medication': 0.2,
                'prob_no_surgery': 0.5
            },
            'tb': {
                'examination_type': 'X-ray',
                'body_region': 'Chest',
                'presenting_symptoms': {'Tuberculosis': ['Cough'], 'Normal': ['Screening']},
                'current_medications': ['None'],
                'previous_surgeries': ['None'],
                'urgency_levels': {'Routine': 1.0}
            },
            'lung_cancer': {
                'examination_type': 'CT',
                'body_region': 'Chest',
                'presenting_symptoms': {'adenocarcinoma': ['Cough'], 'normal': ['Screening']},
                'current_medications': ['None'],
                'previous_surgeries': ['None'],
                'urgency_levels': {'Routine': 1.0}
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump(mock_config, f)
            temp_config_path = f.name
        
        try:
            pipeline = SyntheticDataPipeline(temp_config_path)
            image_files = pipeline.get_image_files('data/test_preprocessed/lung_cancer_ct_scan')
            
            # Should have 9 images total
            assert len(image_files) == 9
        finally:
            os.unlink(temp_config_path)
    
    def test_generate_tb_data(self, test_data_setup):
        """Test generating synthetic data for tuberculosis test images."""
        from scripts.data_preprocessing.baseline_synthetic_data_generator import TBDataGenerator
        
        config = {
            'output': {'format': 'csv', 'random_seed': 42, 'include_image_path': True},
            'faker': {'locale': 'en_US'},
            'demographics': {
                'age_range': {'min': 22, 'max': 85},
                'weight_range': {'min': 45, 'max': 120},
                'height_range': {'min': 150, 'max': 195},
                'gender_distribution': {'Male': 0.48, 'Female': 0.48, 'Other': 0.04}
            },
            'general': {
                'max_symptoms_per_patient': 3,
                'max_medications_per_patient': 3,
                'max_surgeries_per_patient': 2,
                'prob_no_medication': 0.2,
                'prob_no_surgery': 0.5
            },
            'tb': {
                'examination_type': 'X-ray',
                'body_region': 'Chest',
                'presenting_symptoms': {
                    'Tuberculosis': ['Persistent cough', 'Coughing up blood'],
                    'Normal': ['Routine screening']
                },
                'current_medications': ['Isoniazid (TB treatment)', 'None'],
                'previous_surgeries': ['None', 'Lung biopsy'],
                'urgency_levels': {'Routine': 0.6, 'Urgent': 0.3, 'Emergent': 0.1}
            }
        }
        
        generator = TBDataGenerator(config)
        
        # Generate records for test Patient IDs
        records = []
        for patient_id, split, class_name in test_data_setup['tb_patient_ids'][:3]:
            image_path = f"data/test_preprocessed/tb/{split}/{class_name}/{patient_id}.jpg"
            record = generator.generate_patient_record(image_path, class_name)
            records.append(record)
        
        # Verify records
        assert len(records) == 3
        
        for i, record in enumerate(records):
            patient_id, split, class_name = test_data_setup['tb_patient_ids'][i]
            assert record['Patient_ID'] == patient_id
            assert record['Examination_Type'] == 'X-ray'
            assert record['Body_Region'] == 'Chest'
            assert record['Diagnosis_Class'] == class_name
    
    def test_csv_output(self, test_data_setup, tmp_path):
        """Test CSV output generation."""
        from scripts.data_preprocessing.baseline_synthetic_data_generator import (
            TBDataGenerator,
            SyntheticDataPipeline
        )
        
        config = {
            'output': {'format': 'csv', 'random_seed': 42, 'include_image_path': True},
            'faker': {'locale': 'en_US'},
            'demographics': {
                'age_range': {'min': 22, 'max': 85},
                'weight_range': {'min': 45, 'max': 120},
                'height_range': {'min': 150, 'max': 195},
                'gender_distribution': {'Male': 0.48, 'Female': 0.48, 'Other': 0.04}
            },
            'general': {
                'max_symptoms_per_patient': 3,
                'max_medications_per_patient': 3,
                'max_surgeries_per_patient': 2,
                'prob_no_medication': 0.2,
                'prob_no_surgery': 0.5
            },
            'tb': {
                'examination_type': 'X-ray',
                'body_region': 'Chest',
                'presenting_symptoms': {
                    'Tuberculosis': ['Persistent cough'],
                    'Normal': ['Routine screening']
                },
                'current_medications': ['None'],
                'previous_surgeries': ['None'],
                'urgency_levels': {'Routine': 1.0}
            },
            'lung_cancer': {
                'examination_type': 'CT',
                'body_region': 'Chest',
                'presenting_symptoms': {
                    'adenocarcinoma': ['Cough'],
                    'normal': ['Screening']
                },
                'current_medications': ['None'],
                'previous_surgeries': ['None'],
                'urgency_levels': {'Routine': 1.0}
            }
        }
        
        generator = TBDataGenerator(config)
        
        # Create temp config file in /tmp (not mounted volume)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump(config, f)
            temp_config_path = f.name
        
        try:
            pipeline = SyntheticDataPipeline(temp_config_path)
            
            # Generate a few test records
            records = []
            for patient_id, split, class_name in test_data_setup['tb_patient_ids'][:2]:
                image_path = f"data/test_preprocessed/tb/{split}/{class_name}/{patient_id}.jpg"
                record = generator.generate_patient_record(image_path, class_name)
                records.append(record)
            
            # Save to CSV (disable partitioning for test)
            output_file = tmp_path / "test_output.csv"
            pipeline.save_records_csv(records, str(output_file), use_partitioning=False)
            
            # Verify CSV was created
            assert output_file.exists()
            
            # Read and verify CSV contents
            with open(output_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                csv_records = list(reader)
            
            assert len(csv_records) == 2
            
            # Verify headers
            expected_headers = [
                'Patient_Full_Name', 'Patient_ID', 'Presenting_Symptoms',
                'Current_Medications', 'Previous_Relevant_Surgeries',
                'Age_Years', 'Weight_KG', 'Height_CM', 'Gender',
                'Examination_Type', 'Body_Region', 'Urgency_Level',
                'Image_Path', 'Diagnosis_Class'
            ]
            
            for header in expected_headers:
                assert header in reader.fieldnames
        
        finally:
            # Cleanup temp file
            os.unlink(temp_config_path)


# Run tests
if __name__ == '__main__':
    pytest.main([__file__, '-v'])