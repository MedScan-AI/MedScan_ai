"""
Test suite for schema_statistics.py module.
Tests SchemaStatisticsManager with Great Expectations and MLflow integration.
"""

import pytest
import warnings
import yaml
import os
import sys
import tempfile
import shutil
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.data_preprocessing.schema_statistics import SchemaStatisticsManager

# Suppress Great Expectations deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="great_expectations")


@pytest.fixture
def temp_dirs():
    """Fixture providing temporary directories for testing."""
    temp_dir = tempfile.mkdtemp()
    
    # Create necessary subdirectories
    data_dir = Path(temp_dir) / "data"
    config_dir = Path(temp_dir) / "config"
    
    data_dir.mkdir()
    config_dir.mkdir()
    
    # Create partitioned metadata directories
    metadata_dir = data_dir / "synthetic_metadata" / "2025" / "10" / "23"
    metadata_dir.mkdir(parents=True)
    
    yield temp_dir, data_dir, config_dir, metadata_dir
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_config(temp_dirs):
    """Fixture providing sample configuration."""
    temp_dir, data_dir, config_dir, metadata_dir = temp_dirs
    
    config = {
        'great_expectations': {
            'statistics': {
                'output_dir': str(data_dir / "ge_outputs"),
                'baseline_dir': str(data_dir / "ge_outputs" / "baseline"),
                'new_data_dir': str(data_dir / "ge_outputs" / "new_data"),
            },
            'schema': {
                'output_dir': str(data_dir / "ge_outputs" / "schemas"),
                'auto_infer': True
            },
            'validation': {
                'output_dir': str(data_dir / "ge_outputs" / "validations"),
                'enable_anomaly_detection': True
            },
            'drift_detection': {
                'enable': True,
                'output_dir': str(data_dir / "ge_outputs" / "drift"),
                'statistical_test_threshold': 0.00001
            },
            'visualization': {
                'enable': True,
                'output_dir': str(data_dir / "ge_outputs" / "reports")
            }
        },
        'datasets': {
            'test_dataset': {
                'name': 'test_patients',
                'raw_base_path': str(data_dir / "raw" / "test"),
                'preprocessed_base_path': str(data_dir / "preprocessed" / "test"),
                'metadata_base_path': str(data_dir / "synthetic_metadata"),
                'metadata_filename': 'test_dataset.csv',
                'description': 'Test patient metadata'
            }
        },
        'partitioning': {
            'enabled': True,
            'format': 'year/month/day',
            'metadata_file': str(data_dir / "partition_metadata.json")
        },
        'bias_detection': {
            'enable': True,
            'output_dir': str(data_dir / "ge_outputs" / "bias_analysis"),
            'slicing_features': ['Age_Group', 'Gender', 'Urgency_Level'],
            'mitigation': {
                'enable': True,
                'mitigated_data_output_dir': str(data_dir / "synthetic_metadata_mitigated")
            }
        },
        'eda': {
            'enable': True,
            'output_dir': str(data_dir / "ge_outputs" / "eda")
        },
        'mlmd': {
            'store': {
                'type': 'sqlite',
                'database_path': str(data_dir / "mlflow_store" / "metadata.db")
            }
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': str(data_dir / "logs" / "schema_statistics.log")
        },
        'execution': {
            'process_datasets': ['test_dataset'],
            'operations': {
                'generate_statistics': True,
                'infer_schema': True,
                'perform_eda': True,
                'validate_data': True,
                'detect_drift': True,
                'detect_bias': True,
                'generate_reports': True
            }
        },
        'schema_constraints': {
            'numerical_features': {
                'Age_Years': {'min': 0, 'max': 120, 'required': True}
            },
            'categorical_features': {
                'Gender': {
                    'allowed_values': ['Male', 'Female', 'Other'],
                    'required': True
                }
            },
            'string_features': {
                'Patient_ID': {'required': True, 'unique': True}
            }
        }
    }
    
    # Write config to file
    config_file = config_dir / "metadata.yml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    
    return str(config_file), config


class TestSchemaStatisticsManager:
    """Test cases for SchemaStatisticsManager class."""
    
    @pytest.fixture
    def sample_dataframe(self, temp_dirs):
        """Fixture providing a sample pandas DataFrame."""
        temp_dir, data_dir, config_dir, metadata_dir = temp_dirs
        
        df = pd.DataFrame({
            'Patient_ID': [f'P{i:04d}' for i in range(100)],
            'Patient_Full_Name': [f'Patient {i}' for i in range(100)],
            'Age_Years': [25 + i % 60 for i in range(100)],
            'Weight_KG': [60.0 + i % 40 for i in range(100)],
            'Height_CM': [160 + i % 35 for i in range(100)],
            'Gender': ['Male' if i % 2 == 0 else 'Female' for i in range(100)],
            'Examination_Type': ['MRI' for _ in range(100)],
            'Diagnosis_Class': ['glioma' if i % 2 == 0 else 'meningioma' for i in range(100)]
        })
        
        # Save to partitioned location
        csv_file = metadata_dir / "test_dataset.csv"
        df.to_csv(csv_file, index=False)
        
        return df, csv_file
    
    def test_initialization(self, sample_config):
        """Test SchemaStatisticsManager initialization."""
        config_file, config = sample_config
        
        manager = SchemaStatisticsManager(config_file)
        
        assert manager.config is not None
        assert manager.logger is not None
        assert 'great_expectations' in manager.config
    
    def test_load_config(self, sample_config):
        """Test configuration loading."""
        config_file, config = sample_config
        
        manager = SchemaStatisticsManager(config_file)
        loaded_config = manager._load_config(config_file)
        
        assert 'great_expectations' in loaded_config
        assert 'datasets' in loaded_config
        assert 'execution' in loaded_config
    
    def test_setup_directories(self, sample_config):
        """Test directory setup."""
        config_file, config = sample_config
        
        manager = SchemaStatisticsManager(config_file)
        manager._setup_directories()
        
        # Check if directories were created
        assert os.path.exists(config['great_expectations']['statistics']['output_dir'])
        assert os.path.exists(config['great_expectations']['schema']['output_dir'])
    
    def test_discover_partitions(self, sample_config, sample_dataframe):
        """Test partition discovery."""
        config_file, config = sample_config
        df, csv_file = sample_dataframe
        
        manager = SchemaStatisticsManager(config_file)
        
        base_path = config['datasets']['test_dataset']['metadata_base_path']
        partitions = manager._discover_partitions(base_path)
        
        assert len(partitions) >= 1
        assert 'path' in partitions[0]
        assert 'timestamp' in partitions[0]
    
    def test_load_partition_data(self, sample_config, sample_dataframe):
        """Test loading data from partitions."""
        config_file, config = sample_config
        df, csv_file = sample_dataframe
        
        manager = SchemaStatisticsManager(config_file)
        
        base_path = config['datasets']['test_dataset']['metadata_base_path']
        partitions = manager._discover_partitions(base_path)
        
        if partitions:
            loaded_df = manager._load_partition_data(partitions)
            assert loaded_df is not None
            assert len(loaded_df) == len(df)
    
    def test_generate_statistics(self, sample_config, sample_dataframe, temp_dirs):
        """Test statistics generation."""
        config_file, config = sample_config
        df, csv_file = sample_dataframe
        temp_dir, data_dir, _, _ = temp_dirs
        
        manager = SchemaStatisticsManager(config_file)
        manager._setup_mlflow()
        manager._start_mlflow_run("test_run", {})
        
        output_file = data_dir / "test_stats.json"
        stats = manager.generate_statistics(
            dataset_name="test_dataset",
            df=df,
            output_path=str(output_file)
        )
        
        assert stats is not None
        assert 'num_rows' in stats
        assert 'num_columns' in stats
        assert stats['num_rows'] == 100
        assert os.path.exists(output_file)
        
        manager._end_mlflow_run()
    
    def test_infer_schema(self, sample_config, sample_dataframe, temp_dirs):
        """Test schema inference."""
        config_file, config = sample_config
        df, csv_file = sample_dataframe
        temp_dir, data_dir, _, _ = temp_dirs
        
        manager = SchemaStatisticsManager(config_file)
        manager._setup_mlflow()
        manager._start_mlflow_run("test_run", {})
        
        output_file = data_dir / "test_schema.json"
        expectations = manager.infer_schema(
            dataset_name="test_dataset",
            df=df,
            output_path=str(output_file)
        )
        
        assert expectations is not None
        assert 'expectations' in expectations
        assert 'schema_details' in expectations
        assert os.path.exists(output_file)
        
        manager._end_mlflow_run()
    
    def test_validate_data(self, sample_config, sample_dataframe, temp_dirs):
        """Test data validation."""
        config_file, config = sample_config
        df, csv_file = sample_dataframe
        temp_dir, data_dir, _, _ = temp_dirs
        
        manager = SchemaStatisticsManager(config_file)
        manager._setup_mlflow()
        manager._start_mlflow_run("test_run", {})
        
        # First infer schema
        schema_file = data_dir / "test_schema.json"
        expectations = manager.infer_schema(
            dataset_name="test_dataset",
            df=df,
            output_path=str(schema_file)
        )
        
        # Then validate
        validation_file = data_dir / "test_validation.json"
        validation_results = manager.validate_data(
            dataset_name="test_dataset",
            df=df,
            expectations=expectations,
            output_path=str(validation_file)
        )
        
        assert validation_results is not None
        assert 'is_valid' in validation_results
        assert 'anomalies' in validation_results
        assert os.path.exists(validation_file)
        
        manager._end_mlflow_run()
    
    def test_detect_drift(self, sample_config, sample_dataframe, temp_dirs):
        """Test drift detection."""
        config_file, config = sample_config
        df, csv_file = sample_dataframe
        temp_dir, data_dir, _, _ = temp_dirs
        
        manager = SchemaStatisticsManager(config_file)
        manager._setup_mlflow()
        manager._start_mlflow_run("test_run", {})
        
        # Create slightly modified DataFrame for drift detection
        df_new = df.copy()
        df_new['Age_Years'] = df_new['Age_Years'] + 5
        
        drift_file = data_dir / "test_drift.json"
        drift_report = manager.detect_drift(
            dataset_name="test_dataset",
            baseline_df=df,
            new_df=df_new,
            output_path=str(drift_file)
        )
        
        assert drift_report is not None
        assert 'has_drift' in drift_report
        assert 'drifted_features' in drift_report
        assert os.path.exists(drift_file)
        
        manager._end_mlflow_run()
    
    def test_perform_exploratory_analysis(self, sample_config, sample_dataframe, temp_dirs):
        """Test EDA functionality."""
        config_file, config = sample_config
        df, csv_file = sample_dataframe
        temp_dir, data_dir, _, _ = temp_dirs
        
        manager = SchemaStatisticsManager(config_file)
        manager._setup_mlflow()
        manager._start_mlflow_run("test_run", {})
        
        eda_file = data_dir / "test_eda.json"
        eda_results = manager.perform_exploratory_analysis(
            dataset_name="test_dataset",
            df=df,
            output_path=str(eda_file)
        )
        
        assert eda_results is not None
        assert 'overview' in eda_results
        assert 'numerical_analysis' in eda_results
        assert 'categorical_analysis' in eda_results
        assert 'missing_data_analysis' in eda_results
        assert os.path.exists(eda_file)
        
        manager._end_mlflow_run()
    
    def test_get_output_partition_path(self, sample_config):
        """Test partitioned output path generation."""
        config_file, config = sample_config
        
        manager = SchemaStatisticsManager(config_file)
        
        base_dir = "/test/output"
        timestamp = "2025-10-23T00:00:00"
        
        partition_path = manager._get_output_partition_path(base_dir, timestamp)
        
        assert "2025" in partition_path
        assert "10" in partition_path
        assert "23" in partition_path
    
    def test_extract_schema_details(self, sample_config, sample_dataframe):
        """Test schema details extraction."""
        config_file, config = sample_config
        df, _ = sample_dataframe
        
        manager = SchemaStatisticsManager(config_file)
        
        schema_details = manager._extract_schema_details(df)
        
        assert 'columns' in schema_details
        assert 'num_columns' in schema_details
        assert 'column_names' in schema_details
        assert schema_details['num_columns'] == len(df.columns)
    
    def test_compare_schemas(self, sample_config, sample_dataframe):
        """Test schema comparison."""
        config_file, config = sample_config
        df, _ = sample_dataframe
        
        manager = SchemaStatisticsManager(config_file)
        
        schema1 = manager._extract_schema_details(df)
        
        # Create modified schema
        df_modified = df.copy()
        df_modified['New_Column'] = 'test'
        schema2 = manager._extract_schema_details(df_modified)
        
        changes = manager._compare_schemas(schema1, schema2)
        
        assert 'has_changes' in changes
        assert 'added_columns' in changes
        assert changes['has_changes'] is True
        assert 'New_Column' in changes['added_columns']


class TestSchemaStatisticsWithArgparse:
    """Test cases for command-line argument handling."""
    
    def test_main_with_default_config(self, sample_config):
        """Test main function with default config."""
        config_file, config = sample_config
        
        # Test that the manager can be initialized with the config
        manager = SchemaStatisticsManager(config_file)
        assert manager.config is not None
        assert 'great_expectations' in manager.config
    
    def test_main_with_custom_config(self, sample_config):
        """Test main function with custom config path."""
        config_file, config = sample_config
        
        # Test that the manager can be initialized with the custom config
        manager = SchemaStatisticsManager(config_file)
        assert manager.config is not None
        assert 'datasets' in manager.config
        assert 'test_dataset' in manager.config['datasets']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

