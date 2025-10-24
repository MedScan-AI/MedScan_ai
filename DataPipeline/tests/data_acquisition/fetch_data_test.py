"""
Test suite for data acquisition module.
Tests the KaggleDataFetcher and DataAcquisitionPipeline classes.
"""

import pytest
import yaml
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, mock_open
import tempfile
import os

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../scripts/data_acquisition'))

from fetch_data import (
    DataFetcher,
    KaggleDataFetcher,
    DataAcquisitionPipeline
)


class TestKaggleDataFetcher:
    """Test cases for KaggleDataFetcher class."""
    
    @pytest.fixture
    def sample_config(self):
        """Fixture providing sample configuration for tests."""
        return {
            'datasets': [
                {
                    'name': 'lung_cancer_ct_scan',
                    'dataset_id': 'dishantrathi20/ct-scan-images-for-lung-cancer',
                    'download_path': 'data/raw/lung_cancer_ct_scan',
                    'description': 'CT scan images for lung cancer detection'
                },
                {
                    'name': 'tb',
                    'dataset_id': 'tawsifurrahman/tuberculosis-tb-chest-xray-dataset',
                    'download_path': 'data/raw/tb',
                    'description': 'Tuberculosis TB Chest X-ray dataset'
                }
            ],
            'api': {
                'use_kagglehub': True,
                'unzip': True,
                'force_download': False
            }
        }
    
    @pytest.fixture
    def kaggle_fetcher(self, sample_config):
        """Fixture providing KaggleDataFetcher instance."""
        return KaggleDataFetcher(sample_config)
    
    def test_initialization(self, sample_config):
        """Test KaggleDataFetcher initialization."""
        fetcher = KaggleDataFetcher(sample_config)
        
        assert fetcher.config == sample_config
        assert len(fetcher.datasets) == 2
        assert fetcher.use_kagglehub is True
        assert fetcher.api_config['unzip'] is True
    
    def test_initialization_with_empty_config(self):
        """Test initialization with minimal configuration."""
        fetcher = KaggleDataFetcher({})
        
        assert fetcher.datasets == []
        assert fetcher.api_config == {}
        assert fetcher.use_kagglehub is True  # Default value
    
    @patch('builtins.__import__')
    def test_validate_success_with_kagglehub(self, mock_import, kaggle_fetcher):
        """Test successful validation with kagglehub."""
        # Mock successful kagglehub import
        def import_mock(name, *args, **kwargs):
            if name == 'kagglehub':
                return MagicMock()
            return __import__(name, *args, **kwargs)
        
        mock_import.side_effect = import_mock
        result = kaggle_fetcher.validate()
        
        assert result is True
    
    @patch('builtins.__import__')
    def test_validate_failure_missing_kagglehub(self, mock_import, kaggle_fetcher):
        """Test validation failure when kagglehub is not installed."""
        # Mock failed kagglehub import
        def import_mock(name, *args, **kwargs):
            if name == 'kagglehub':
                raise ImportError("No module named 'kagglehub'")
            return __import__(name, *args, **kwargs)
        
        mock_import.side_effect = import_mock
        result = kaggle_fetcher.validate()
        
        assert result is False
    
    @patch('builtins.__import__')
    def test_validate_failure_no_datasets(self, mock_import, sample_config):
        """Test validation failure when no datasets are configured."""
        sample_config['datasets'] = []
        fetcher = KaggleDataFetcher(sample_config)
        
        # Mock successful kagglehub import
        def import_mock(name, *args, **kwargs):
            if name == 'kagglehub':
                return MagicMock()
            return __import__(name, *args, **kwargs)
        
        mock_import.side_effect = import_mock
        result = fetcher.validate()
        
        assert result is False
    
    @patch('builtins.__import__')
    def test_validate_with_kaggle_api_success(self, mock_import, sample_config):
        """Test validation with Kaggle API when credentials exist."""
        sample_config['api']['use_kagglehub'] = False
        fetcher = KaggleDataFetcher(sample_config)
        
        # Mock kaggle import and Path
        mock_kaggle_json = MagicMock()
        mock_kaggle_json.exists.return_value = True
        
        def import_mock(name, *args, **kwargs):
            if name == 'kaggle':
                return MagicMock()
            return __import__(name, *args, **kwargs)
        
        mock_import.side_effect = import_mock
        
        with patch('fetch_data.Path') as mock_path:
            mock_path.home.return_value.__truediv__.return_value.__truediv__.return_value = mock_kaggle_json
            result = fetcher.validate()
        
        assert result is True
    
    @patch('builtins.__import__')
    def test_validate_with_kaggle_api_missing_credentials(self, mock_import, sample_config):
        """Test validation failure when kaggle.json is missing."""
        sample_config['api']['use_kagglehub'] = False
        fetcher = KaggleDataFetcher(sample_config)
        
        # Mock kaggle import and Path
        mock_kaggle_json = MagicMock()
        mock_kaggle_json.exists.return_value = False
        
        def import_mock(name, *args, **kwargs):
            if name == 'kaggle':
                return MagicMock()
            return __import__(name, *args, **kwargs)
        
        mock_import.side_effect = import_mock
        
        with patch('fetch_data.Path') as mock_path:
            mock_path.home.return_value.__truediv__.return_value.__truediv__.return_value = mock_kaggle_json
            result = fetcher.validate()
        
        assert result is False
    
    def test_check_existing_data_with_existing_directory(self, kaggle_fetcher):
        """Test _check_existing_data when data exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file in the directory
            test_file = Path(tmpdir) / 'test.txt'
            test_file.write_text('test data')
            
            result = kaggle_fetcher._check_existing_data(tmpdir)
            
            assert result is True
    
    def test_check_existing_data_with_empty_directory(self, kaggle_fetcher):
        """Test _check_existing_data with empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = kaggle_fetcher._check_existing_data(tmpdir)
            
            assert result is False
    
    def test_check_existing_data_with_nonexistent_path(self, kaggle_fetcher):
        """Test _check_existing_data with non-existent path."""
        result = kaggle_fetcher._check_existing_data('/nonexistent/path')
        
        assert result is False
    
    @patch('builtins.__import__')
    @patch('fetch_data.Path')
    def test_fetch_with_kagglehub(self, mock_path, mock_import, kaggle_fetcher):
        """Test fetching datasets using kagglehub."""
        # Mock directory creation
        mock_path.return_value.mkdir = MagicMock()
        
        # Mock kagglehub module
        mock_kagglehub = MagicMock()
        mock_kagglehub.dataset_download.return_value = '/path/to/downloaded/data'
        
        def import_mock(name, *args, **kwargs):
            if name == 'kagglehub':
                return mock_kagglehub
            return __import__(name, *args, **kwargs)
        
        mock_import.side_effect = import_mock
        
        # Mock validation
        with patch.object(kaggle_fetcher, 'validate', return_value=True):
            # Mock existing data check and copy method
            with patch.object(kaggle_fetcher, '_check_existing_data', return_value=False):
                with patch.object(kaggle_fetcher, '_copy_data_to_location'):
                    paths = kaggle_fetcher.fetch()
        
        assert len(paths) == 2
        assert mock_kagglehub.dataset_download.call_count == 2
    
    @patch('builtins.__import__')
    @patch('fetch_data.Path')
    def test_fetch_specific_dataset(self, mock_path, mock_import, kaggle_fetcher):
        """Test fetching a specific dataset by name."""
        # Mock directory creation
        mock_path.return_value.mkdir = MagicMock()
        
        # Mock kagglehub module
        mock_kagglehub = MagicMock()
        mock_kagglehub.dataset_download.return_value = '/path/to/downloaded/data'
        
        def import_mock(name, *args, **kwargs):
            if name == 'kagglehub':
                return mock_kagglehub
            return __import__(name, *args, **kwargs)
        
        mock_import.side_effect = import_mock
        
        # Mock validation
        with patch.object(kaggle_fetcher, 'validate', return_value=True):
            # Mock existing data check and copy method
            with patch.object(kaggle_fetcher, '_check_existing_data', return_value=False):
                with patch.object(kaggle_fetcher, '_copy_data_to_location'):
                    paths = kaggle_fetcher.fetch(dataset_name='lung_cancer_ct_scan')
        
        assert len(paths) == 1
        assert mock_kagglehub.dataset_download.call_count == 1
        mock_kagglehub.dataset_download.assert_called_with(
            'dishantrathi20/ct-scan-images-for-lung-cancer'
        )
    
    def test_fetch_nonexistent_dataset(self, kaggle_fetcher):
        """Test fetching a dataset that doesn't exist in config."""
        with patch.object(kaggle_fetcher, 'validate', return_value=True):
            paths = kaggle_fetcher.fetch(dataset_name='nonexistent_dataset')
        
        assert paths == []
    
    def test_fetch_validation_failure(self, kaggle_fetcher):
        """Test fetch when validation fails."""
        with patch.object(kaggle_fetcher, 'validate', return_value=False):
            with pytest.raises(RuntimeError, match="Validation failed"):
                kaggle_fetcher.fetch()
    
    @patch('builtins.__import__')
    @patch('fetch_data.Path')
    def test_fetch_with_existing_data_skip_download(self, mock_path, mock_import, kaggle_fetcher):
        """Test that existing data is not re-downloaded when force_download is False."""
        # Mock directory creation
        mock_path.return_value.mkdir = MagicMock()
        
        # Mock kagglehub module
        mock_kagglehub = MagicMock()
        
        def import_mock(name, *args, **kwargs):
            if name == 'kagglehub':
                return mock_kagglehub
            return __import__(name, *args, **kwargs)
        
        mock_import.side_effect = import_mock
        
        # Mock validation
        with patch.object(kaggle_fetcher, 'validate', return_value=True):
            # Mock existing data check - data exists
            with patch.object(kaggle_fetcher, '_check_existing_data', return_value=True):
                paths = kaggle_fetcher.fetch()
        
        # Should return paths but not call kagglehub
        assert len(paths) == 2
        assert mock_kagglehub.dataset_download.call_count == 0
    
    @patch('builtins.__import__')
    @patch('fetch_data.Path')
    def test_fetch_with_force_download(self, mock_path, mock_import, sample_config):
        """Test that data is re-downloaded when force_download is True."""
        sample_config['api']['force_download'] = True
        fetcher = KaggleDataFetcher(sample_config)
        
        # Mock directory creation
        mock_path.return_value.mkdir = MagicMock()
        
        # Mock kagglehub module
        mock_kagglehub = MagicMock()
        mock_kagglehub.dataset_download.return_value = '/path/to/downloaded/data'
        
        def import_mock(name, *args, **kwargs):
            if name == 'kagglehub':
                return mock_kagglehub
            return __import__(name, *args, **kwargs)
        
        mock_import.side_effect = import_mock
        
        # Mock validation
        with patch.object(fetcher, 'validate', return_value=True):
            # Even if data exists, should download
            with patch.object(fetcher, '_check_existing_data', return_value=True):
                with patch.object(fetcher, '_copy_data_to_location'):
                    paths = fetcher.fetch()
        
        # Should still download despite existing data
        assert len(paths) == 2
        assert mock_kagglehub.dataset_download.call_count == 2
    
    @patch('builtins.__import__')
    @patch('fetch_data.Path')
    def test_fetch_with_kaggle_api(self, mock_path, mock_import, sample_config):
        """Test fetching using Kaggle API instead of kagglehub."""
        sample_config['api']['use_kagglehub'] = False
        fetcher = KaggleDataFetcher(sample_config)
        
        # Mock directory creation
        mock_path.return_value.mkdir = MagicMock()
        
        # Mock Kaggle API
        mock_api_instance = MagicMock()
        mock_kaggle_module = MagicMock()
        mock_kaggle_module.api.kaggle_api_extended.KaggleApi.return_value = mock_api_instance
        
        def import_mock(name, *args, **kwargs):
            if name == 'kaggle':
                return mock_kaggle_module
            if name == 'kaggle.api.kaggle_api_extended':
                mock_extended = MagicMock()
                mock_extended.KaggleApi = MagicMock(return_value=mock_api_instance)
                return mock_extended
            return __import__(name, *args, **kwargs)
        
        mock_import.side_effect = import_mock
        
        # Mock validation
        mock_kaggle_json = MagicMock()
        mock_kaggle_json.exists.return_value = True
        mock_path.home.return_value.__truediv__.return_value.__truediv__.return_value = mock_kaggle_json
        
        with patch.object(fetcher, '_check_existing_data', return_value=False):
            paths = fetcher.fetch()
        
        assert len(paths) == 2
        assert mock_api_instance.authenticate.call_count == 2
        assert mock_api_instance.dataset_download_files.call_count == 2
    
    @patch('builtins.__import__')
    @patch('fetch_data.Path')
    def test_fetch_handles_individual_dataset_failure(self, mock_path, mock_import, kaggle_fetcher):
        """Test that one dataset failure doesn't stop others from downloading."""
        # Mock directory creation
        mock_path.return_value.mkdir = MagicMock()
        
        # Mock kagglehub module
        mock_kagglehub = MagicMock()
        # Mock kagglehub to fail on first call, succeed on second
        mock_kagglehub.dataset_download.side_effect = [
            Exception("Download failed"),
            '/path/to/downloaded/data'
        ]
        
        def import_mock(name, *args, **kwargs):
            if name == 'kagglehub':
                return mock_kagglehub
            return __import__(name, *args, **kwargs)
        
        mock_import.side_effect = import_mock
        
        with patch.object(kaggle_fetcher, 'validate', return_value=True):
            with patch.object(kaggle_fetcher, '_check_existing_data', return_value=False):
                with patch.object(kaggle_fetcher, '_copy_data_to_location'):
                    paths = kaggle_fetcher.fetch()
        
        # Should have one successful download
        assert len(paths) == 1


class TestDataAcquisitionPipeline:
    """Test cases for DataAcquisitionPipeline class."""
    
    @pytest.fixture
    def sample_yaml_config(self):
        """Fixture providing sample YAML configuration."""
        return """
data_acquisition:
  kaggle:
    datasets:
      - name: "test_dataset"
        dataset_id: "test/dataset"
        download_path: "data/raw/test"
        description: "Test dataset"
    api:
      use_kagglehub: true
      unzip: true
      force_download: false
"""
    
    @pytest.fixture
    def config_file(self, sample_yaml_config):
        """Fixture providing temporary config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write(sample_yaml_config)
            config_path = f.name
        
        yield config_path
        
        # Cleanup
        os.unlink(config_path)
    
    def test_initialization(self, config_file):
        """Test DataAcquisitionPipeline initialization."""
        pipeline = DataAcquisitionPipeline(config_file)
        
        assert pipeline.config_path == config_file
        assert 'data_acquisition' in pipeline.config
        assert 'kaggle' in pipeline.fetchers
        assert isinstance(pipeline.fetchers['kaggle'], KaggleDataFetcher)
    
    def test_load_config(self, config_file):
        """Test configuration loading."""
        pipeline = DataAcquisitionPipeline(config_file)
        config = pipeline._load_config()
        
        assert 'data_acquisition' in config
        assert 'kaggle' in config['data_acquisition']
    
    def test_load_config_file_not_found(self):
        """Test loading non-existent config file."""
        with pytest.raises(FileNotFoundError):
            pipeline = DataAcquisitionPipeline('/nonexistent/config.yml')
    
    def test_initialize_fetchers(self, config_file):
        """Test fetcher initialization."""
        pipeline = DataAcquisitionPipeline(config_file)
        fetchers = pipeline._initialize_fetchers()
        
        assert 'kaggle' in fetchers
        assert isinstance(fetchers['kaggle'], KaggleDataFetcher)
    
    def test_run_all_fetchers(self, config_file):
        """Test running pipeline with all fetchers."""
        pipeline = DataAcquisitionPipeline(config_file)
        
        # Mock the fetch method
        with patch.object(pipeline.fetchers['kaggle'], 'fetch', return_value=['/path/to/data']) as mock_fetch:
            pipeline.run()
        
        mock_fetch.assert_called_once()
    
    def test_run_specific_source(self, config_file):
        """Test running pipeline with specific source."""
        pipeline = DataAcquisitionPipeline(config_file)
        
        with patch.object(pipeline.fetchers['kaggle'], 'fetch', return_value=['/path/to/data']) as mock_fetch:
            pipeline.run(source='kaggle')
        
        mock_fetch.assert_called_once()
    
    def test_run_invalid_source(self, config_file):
        """Test running pipeline with invalid source."""
        pipeline = DataAcquisitionPipeline(config_file)
        
        with patch.object(pipeline.fetchers['kaggle'], 'fetch') as mock_fetch:
            pipeline.run(source='invalid_source')
        
        # Should not call fetch for invalid source
        mock_fetch.assert_not_called()
    
    def test_run_with_dataset_name(self, config_file):
        """Test running pipeline with specific dataset name."""
        pipeline = DataAcquisitionPipeline(config_file)
        
        with patch.object(pipeline.fetchers['kaggle'], 'fetch', return_value=['/path/to/data']) as mock_fetch:
            pipeline.run(dataset_name='test_dataset')
        
        mock_fetch.assert_called_once_with(dataset_name='test_dataset')
    
    def test_run_handles_fetcher_exception(self, config_file):
        """Test that pipeline handles exceptions from fetchers gracefully."""
        pipeline = DataAcquisitionPipeline(config_file)
        
        with patch.object(pipeline.fetchers['kaggle'], 'fetch', side_effect=Exception("Fetch failed")):
            # Should not raise exception, just log error
            pipeline.run()


class TestDataFetcherAbstractClass:
    """Test cases for DataFetcher abstract base class."""
    
    def test_cannot_instantiate_abstract_class(self):
        """Test that DataFetcher cannot be instantiated directly."""
        with pytest.raises(TypeError):
            fetcher = DataFetcher({})
    
    def test_subclass_must_implement_fetch(self):
        """Test that subclasses must implement fetch method."""
        class IncompleteFetcher(DataFetcher):
            def validate(self):
                return True
        
        with pytest.raises(TypeError):
            fetcher = IncompleteFetcher({})
    
    def test_subclass_must_implement_validate(self):
        """Test that subclasses must implement validate method."""
        class IncompleteFetcher(DataFetcher):
            def fetch(self, **kwargs):
                return ""
        
        with pytest.raises(TypeError):
            fetcher = IncompleteFetcher({})


class TestDataAcquisitionAndStorage:
    """Test cases for data acquisition and storage to configured locations."""
    
    @pytest.fixture
    def temp_config_with_paths(self):
        """Fixture providing config with temporary download paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'data_acquisition': {
                    'kaggle': {
                        'use_partitioning': False,  # Disable partitioning for tests
                        'datasets': [
                            {
                                'name': 'test_dataset_1',
                                'dataset_id': 'test/dataset1',
                                'download_path': os.path.join(tmpdir, 'dataset1'),
                                'description': 'Test dataset 1'
                            },
                            {
                                'name': 'test_dataset_2',
                                'dataset_id': 'test/dataset2',
                                'download_path': os.path.join(tmpdir, 'dataset2'),
                                'description': 'Test dataset 2'
                            }
                        ],
                        'api': {
                            'use_kagglehub': True,
                            'unzip': True,
                            'force_download': False
                        }
                    }
                }
            }
            
            # Create config file
            config_file = os.path.join(tmpdir, 'test_config.yml')
            with open(config_file, 'w') as f:
                yaml.dump(config, f)
            
            yield config_file, tmpdir, config
    
    @patch('builtins.__import__')
    def test_fetch_and_verify_download_location(self, mock_import, temp_config_with_paths):
        """Test that data is fetched and paths are correctly returned."""
        config_file, tmpdir, config = temp_config_with_paths
        
        # Mock kagglehub
        mock_kagglehub = MagicMock()
        mock_kagglehub.dataset_download.side_effect = lambda ds_id: os.path.join(tmpdir, ds_id.split('/')[-1])
        
        def import_mock(name, *args, **kwargs):
            if name == 'kagglehub':
                return mock_kagglehub
            return __import__(name, *args, **kwargs)
        
        mock_import.side_effect = import_mock
        
        # Initialize pipeline
        pipeline = DataAcquisitionPipeline(config_file)
        
        # Run pipeline
        paths = pipeline.run()
        
        # Verify structure
        assert 'kaggle' in paths
        assert len(paths['kaggle']) == 2
        
        # Verify all paths are strings
        for path in paths['kaggle']:
            assert isinstance(path, str)
            assert len(path) > 0
    
    def test_get_dataset_info(self, temp_config_with_paths):
        """Test getting dataset information."""
        config_file, tmpdir, config = temp_config_with_paths
        
        pipeline = DataAcquisitionPipeline(config_file)
        
        # Get dataset info
        info = pipeline.get_dataset_info()
        
        # Verify structure
        assert 'kaggle' in info
        assert len(info['kaggle']) == 2
        
        # Verify dataset info content
        for dataset in info['kaggle']:
            assert 'name' in dataset
            assert 'dataset_id' in dataset
            assert 'description' in dataset
            assert 'configured_path' in dataset
        
        # Verify specific datasets
        names = [d['name'] for d in info['kaggle']]
        assert 'test_dataset_1' in names
        assert 'test_dataset_2' in names
    
    def test_get_dataset_paths(self, temp_config_with_paths):
        """Test getting dataset path mappings."""
        config_file, tmpdir, config = temp_config_with_paths
        
        pipeline = DataAcquisitionPipeline(config_file)
        
        # Get dataset paths
        paths = pipeline.get_data_paths()
        
        # Verify structure
        assert 'kaggle' in paths
        assert isinstance(paths['kaggle'], dict)
    
    @patch('builtins.__import__')
    def test_download_location_matches_config(self, mock_import, temp_config_with_paths):
        """Test that configured download paths are created."""
        config_file, tmpdir, config = temp_config_with_paths
        
        # Get configured paths
        expected_paths = [
            config['data_acquisition']['kaggle']['datasets'][0]['download_path'],
            config['data_acquisition']['kaggle']['datasets'][1]['download_path']
        ]
        
        # Mock kagglehub
        mock_kagglehub = MagicMock()
        mock_kagglehub.dataset_download.side_effect = lambda ds_id: os.path.join(tmpdir, ds_id.split('/')[-1])
        
        def import_mock(name, *args, **kwargs):
            if name == 'kagglehub':
                return mock_kagglehub
            return __import__(name, *args, **kwargs)
        
        mock_import.side_effect = import_mock
        
        # Initialize and run pipeline
        pipeline = DataAcquisitionPipeline(config_file)
        pipeline.run()
        
        # Verify directories were created
        for path in expected_paths:
            assert os.path.exists(path), f"Download directory {path} was not created"
    
    @patch('builtins.__import__')
    def test_returned_paths_accessible_downstream(self, mock_import, temp_config_with_paths):
        """Test that returned paths can be used for downstream processing."""
        config_file, tmpdir, config = temp_config_with_paths
        
        # Mock kagglehub to return actual temp paths
        downloaded_locations = []
        
        def mock_download(ds_id):
            # Create actual directory to simulate download
            download_path = os.path.join(tmpdir, 'downloads', ds_id.split('/')[-1])
            os.makedirs(download_path, exist_ok=True)
            
            # Create a dummy file
            dummy_file = os.path.join(download_path, 'data.txt')
            with open(dummy_file, 'w') as f:
                f.write('test data')
            
            downloaded_locations.append(download_path)
            return download_path
        
        mock_kagglehub = MagicMock()
        mock_kagglehub.dataset_download.side_effect = mock_download
        
        def import_mock(name, *args, **kwargs):
            if name == 'kagglehub':
                return mock_kagglehub
            return __import__(name, *args, **kwargs)
        
        mock_import.side_effect = import_mock
        
        # Run pipeline
        pipeline = DataAcquisitionPipeline(config_file)
        paths = pipeline.run()
        
        # Verify paths are accessible
        assert 'kaggle' in paths
        for path in paths['kaggle']:
            # Path should exist
            assert os.path.exists(path), f"Path {path} does not exist"
            
            # Should be able to list contents
            contents = os.listdir(path)
            assert len(contents) > 0, f"Path {path} is empty"
    
    @patch('builtins.__import__')
    def test_pipeline_returns_dict_with_source_mapping(self, mock_import, temp_config_with_paths):
        """Test that pipeline.run() returns a dict mapping sources to paths."""
        config_file, tmpdir, config = temp_config_with_paths
        
        # Mock kagglehub
        mock_kagglehub = MagicMock()
        mock_kagglehub.dataset_download.return_value = tmpdir
        
        def import_mock(name, *args, **kwargs):
            if name == 'kagglehub':
                return mock_kagglehub
            return __import__(name, *args, **kwargs)
        
        mock_import.side_effect = import_mock
        
        # Run pipeline
        pipeline = DataAcquisitionPipeline(config_file)
        result = pipeline.run()
        
        # Verify return type and structure
        assert isinstance(result, dict), "run() should return a dictionary"
        assert 'kaggle' in result, "Result should contain 'kaggle' key"
        assert isinstance(result['kaggle'], list), "Source value should be a list"
    
    def test_kaggle_fetcher_get_dataset_info_method(self, temp_config_with_paths):
        """Test KaggleDataFetcher's get_dataset_info method."""
        config_file, tmpdir, config = temp_config_with_paths
        
        fetcher_config = config['data_acquisition']['kaggle']
        fetcher = KaggleDataFetcher(fetcher_config)
        
        info = fetcher.get_dataset_info()
        
        # Verify it returns a list
        assert isinstance(info, list)
        assert len(info) == 2
        
        # Verify each item has required fields
        for item in info:
            assert 'name' in item
            assert 'dataset_id' in item
            assert 'description' in item
            assert 'configured_path' in item
    
    def test_kaggle_fetcher_get_dataset_paths_method(self, temp_config_with_paths):
        """Test KaggleDataFetcher's get_dataset_paths method."""
        config_file, tmpdir, config = temp_config_with_paths
        
        fetcher_config = config['data_acquisition']['kaggle']
        fetcher = KaggleDataFetcher(fetcher_config)
        
        paths = fetcher.get_dataset_paths()
        
        # Verify it returns a dictionary
        assert isinstance(paths, dict)
        
        # Should have entries for configured datasets
        assert 'test_dataset_1' in paths
        assert 'test_dataset_2' in paths
    
    @patch('builtins.__import__')
    def test_data_copied_to_configured_location(self, mock_import, temp_config_with_paths):
        """Test that data is copied from kagglehub cache to configured location."""
        config_file, tmpdir, config = temp_config_with_paths
        
        # Create mock kagglehub cache with data
        kagglehub_cache = os.path.join(tmpdir, 'kagglehub_cache')
        os.makedirs(kagglehub_cache, exist_ok=True)
        
        # Create a mock dataset with 5 test images
        for i in range(5):
            test_file = os.path.join(kagglehub_cache, f'image_{i}.jpg')
            with open(test_file, 'w') as f:
                f.write(f'mock image data {i}')
        
        # Mock kagglehub to return our test cache
        mock_kagglehub = MagicMock()
        mock_kagglehub.dataset_download.return_value = kagglehub_cache
        
        def import_mock(name, *args, **kwargs):
            if name == 'kagglehub':
                return mock_kagglehub
            return __import__(name, *args, **kwargs)
        
        mock_import.side_effect = import_mock
        
        # Run pipeline
        pipeline = DataAcquisitionPipeline(config_file)
        paths = pipeline.run()
        
        # Verify data was copied to configured locations
        configured_paths = [
            config['data_acquisition']['kaggle']['datasets'][0]['download_path'],
            config['data_acquisition']['kaggle']['datasets'][1]['download_path']
        ]
        
        for configured_path in configured_paths:
            # Directory should exist
            assert os.path.exists(configured_path), f"Configured path {configured_path} does not exist"
            
            # Should contain the 5 test images
            files = os.listdir(configured_path)
            image_files = [f for f in files if f.endswith('.jpg')]
            assert len(image_files) == 5, f"Expected 5 images in {configured_path}, found {len(image_files)}"
            
            # Verify image file names
            for i in range(5):
                assert f'image_{i}.jpg' in files, f"image_{i}.jpg not found in {configured_path}"
    
    @patch('builtins.__import__')
    def test_actual_image_files_written_to_data_folder(self, mock_import):
        """Test that actual image files are written to data folder with real paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create config with data folder
            data_folder = os.path.join(tmpdir, 'data', 'test_images')
            
            config = {
                'data_acquisition': {
                    'kaggle': {
                        'use_partitioning': False,  # Disable partitioning for tests
                        'datasets': [
                            {
                                'name': 'test_images',
                                'dataset_id': 'test/images',
                                'download_path': data_folder,
                                'description': 'Test images'
                            }
                        ],
                        'api': {
                            'use_kagglehub': True,
                            'unzip': True,
                            'force_download': False
                        }
                    }
                }
            }
            
            # Create config file
            config_file = os.path.join(tmpdir, 'config.yml')
            with open(config_file, 'w') as f:
                yaml.dump(config, f)
            
            # Create mock kagglehub cache with 5 test images
            kagglehub_cache = os.path.join(tmpdir, 'kagglehub_cache')
            os.makedirs(kagglehub_cache, exist_ok=True)
            
            # Write actual test image files (mock JPEG headers)
            for i in range(5):
                img_path = os.path.join(kagglehub_cache, f'test_image_{i+1}.jpg')
                with open(img_path, 'wb') as f:
                    # Write minimal JPEG header
                    f.write(b'\xff\xd8\xff\xe0')
                    f.write(f'FAKE_IMAGE_DATA_{i+1}'.encode())
                    f.write(b'\xff\xd9')
            
            # Mock kagglehub
            mock_kagglehub = MagicMock()
            mock_kagglehub.dataset_download.return_value = kagglehub_cache
            
            def import_mock(name, *args, **kwargs):
                if name == 'kagglehub':
                    return mock_kagglehub
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = import_mock
            
            # Initialize and run pipeline
            pipeline = DataAcquisitionPipeline(config_file)
            paths = pipeline.run()
            
            # Verify the data folder was created
            assert os.path.exists(data_folder), f"Data folder {data_folder} was not created"
            
            # Verify 5 image files exist
            image_files = [f for f in os.listdir(data_folder) if f.endswith('.jpg')]
            assert len(image_files) == 5, f"Expected 5 images, found {len(image_files)}"
            
            # Verify each image file
            for i in range(5):
                img_name = f'test_image_{i+1}.jpg'
                img_path = os.path.join(data_folder, img_name)
                
                # File should exist
                assert os.path.exists(img_path), f"Image {img_name} not found"
                
                # File should have content
                assert os.path.getsize(img_path) > 0, f"Image {img_name} is empty"
                
                # Read and verify content
                with open(img_path, 'rb') as f:
                    content = f.read()
                    assert content.startswith(b'\xff\xd8\xff\xe0'), f"Image {img_name} doesn't have JPEG header"
                    assert f'FAKE_IMAGE_DATA_{i+1}'.encode() in content, f"Image {img_name} missing expected data"
            
            # Verify returned paths point to the data folder
            assert 'kaggle' in paths
            assert data_folder in paths['kaggle']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

