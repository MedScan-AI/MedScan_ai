"""
Data Acquisition Module for Medical Image Datasets
Fetches data from Kaggle datasets using object-oriented design.
"""

import os
import yaml
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from abc import ABC, abstractmethod
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataFetcher(ABC):
    """Abstract base class for data fetchers."""
    
    def __init__(self, config: Dict):
        """
        Initialize the data fetcher with configuration.
        
        Args:
            config: Configuration dictionary for the fetcher
        """
        self.config = config
        
    @abstractmethod
    def fetch(self, **kwargs) -> str:
        """
        Fetch data from the source.
        
        Returns:
            Path to the downloaded data
        """
        pass
    
    @abstractmethod
    def validate(self) -> bool:
        """
        Validate the fetcher configuration and prerequisites.
        
        Returns:
            True if validation passes, False otherwise
        """
        pass


class KaggleDataFetcher(DataFetcher):
    """Fetcher for Kaggle datasets."""
    
    def __init__(self, config: Dict):
        """
        Initialize Kaggle data fetcher.
        
        Args:
            config: Configuration dictionary containing Kaggle settings
        """
        super().__init__(config)
        self.datasets = config.get('datasets', [])
        self.api_config = config.get('api', {})
        self.use_kagglehub = self.api_config.get('use_kagglehub', True)
        self.use_partitioning = config.get('use_partitioning', True)
        
    def validate(self) -> bool:
        """
        Validate Kaggle API credentials and configuration.
        
        Returns:
            True if validation passes, False otherwise
        """
        try:
            if self.use_kagglehub:
                import kagglehub
                logger.info("kagglehub library found")
            else:
                import kaggle
                # Check if kaggle.json exists
                kaggle_json_path = Path.home() / '.kaggle' / 'kaggle.json'
                if not kaggle_json_path.exists():
                    logger.error(
                        "Kaggle API credentials not found. "
                        "Please set up ~/.kaggle/kaggle.json"
                    )
                    return False
            
            if not self.datasets:
                logger.error("No datasets configured")
                return False
                
            logger.info("Validation successful")
            return True
            
        except ImportError as e:
            logger.error(f"Required library not found: {e}")
            logger.error(
                "Please install kagglehub: pip install kagglehub"
            )
            return False
    
    def fetch(self, dataset_name: Optional[str] = None) -> List[str]:
        """
        Fetch datasets from Kaggle.
        
        Args:
            dataset_name: Optional specific dataset name to fetch.
                         If None, fetches all configured datasets.
        
        Returns:
            List of paths to downloaded datasets
        """
        if not self.validate():
            raise RuntimeError("Validation failed. Cannot proceed with fetch.")
        
        downloaded_paths = []
        
        # Filter datasets if specific name provided
        datasets_to_fetch = self.datasets
        if dataset_name:
            datasets_to_fetch = [
                d for d in self.datasets if d.get('name') == dataset_name
            ]
            if not datasets_to_fetch:
                logger.error(f"Dataset '{dataset_name}' not found in configuration")
                return []
        
        for dataset in datasets_to_fetch:
            try:
                path = self._fetch_single_dataset(dataset)
                downloaded_paths.append(path)
            except Exception as e:
                logger.error(
                    f"Failed to fetch dataset {dataset.get('name')}: {e}"
                )
                
        return downloaded_paths
    
    def _fetch_single_dataset(self, dataset: Dict) -> str:
        """
        Fetch a single dataset from Kaggle with optional partitioning.
        
        Args:
            dataset: Dataset configuration dictionary
        
        Returns:
            Path to the downloaded dataset
        """
        dataset_id = dataset.get('dataset_id')
        base_download_path = dataset.get('download_path')
        name = dataset.get('name')
        description = dataset.get('description', '')
        
        logger.info(f"Fetching dataset: {name}")
        logger.info(f"Description: {description}")
        logger.info(f"Dataset ID: {dataset_id}")
        
        # Determine actual download path (with or without partitioning)
        if self.use_partitioning:
            download_path = self._get_partition_path(base_download_path)
            logger.info(f"Using partitioned path: {download_path}")
        else:
            download_path = base_download_path
        
        # Create download directory
        Path(download_path).mkdir(parents=True, exist_ok=True)
        
        # Check if data already exists and force_download is False
        if not self.api_config.get('force_download', False):
            if self._check_existing_data(download_path):
                logger.info(
                    f"Data already exists at {download_path}. "
                    "Skipping download (set force_download: true to override)"
                )
                return download_path
        
        if self.use_kagglehub:
            kagglehub_path = self._fetch_with_kagglehub(dataset_id)
            logger.info(f"Dataset downloaded by kagglehub to: {kagglehub_path}")
            
            # Copy data from kagglehub cache to configured location
            logger.info(f"Copying data to partitioned location: {download_path}")
            self._copy_data_to_location(kagglehub_path, download_path)
            logger.info(f"Data successfully copied to: {download_path}")
            logger.info(f"Data accessible at: {download_path}")
            
            # Return the configured path where data now exists
            return download_path
        else:
            self._fetch_with_kaggle_api(dataset_id, download_path)
            logger.info(f"Data accessible at: {download_path}")
            return download_path
    
    def _fetch_with_kagglehub(self, dataset_id: str) -> str:
        """
        Fetch dataset using kagglehub library.
        
        Args:
            dataset_id: Kaggle dataset identifier
        
        Returns:
            Path to downloaded dataset
        """
        import kagglehub
        
        logger.info(f"Downloading dataset using kagglehub: {dataset_id}")
        path = kagglehub.dataset_download(dataset_id)
        logger.info(f"Dataset downloaded successfully to: {path}")
        
        return path
    
    def _fetch_with_kaggle_api(self, dataset_id: str, download_path: str) -> None:
        """
        Fetch dataset using kaggle API.
        
        Args:
            dataset_id: Kaggle dataset identifier
            download_path: Path where dataset should be downloaded
        """
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        logger.info(f"Downloading dataset using Kaggle API: {dataset_id}")
        
        api = KaggleApi()
        api.authenticate()
        
        # Download dataset
        api.dataset_download_files(
            dataset_id,
            path=download_path,
            unzip=self.api_config.get('unzip', True)
        )
        
        logger.info(f"Dataset downloaded successfully to: {download_path}")
    
    def _copy_data_to_location(self, source_path: str, target_path: str) -> None:
        """
        Copy data from source to target location.
        
        Args:
            source_path: Source directory path (e.g., kagglehub cache)
            target_path: Target directory path (configured location)
        """
        source = Path(source_path)
        target = Path(target_path)
        
        # If target already has data and we're not forcing, skip
        if self._check_existing_data(str(target)) and not self.api_config.get('force_download', False):
            logger.info(f"Target location {target} already has data, skipping copy")
            return
        
        # Remove target if it exists and we're forcing
        if target.exists() and self.api_config.get('force_download', False):
            logger.info(f"Removing existing data at {target}")
            shutil.rmtree(target)
        
        # Create parent directory
        target.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy all contents from source to target
        if source.is_dir():
            logger.info(f"Copying directory contents from {source} to {target}")
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            logger.warning(f"Source path {source} is not a directory")
    
    def _get_partition_path(self, base_path: str, timestamp: datetime = None) -> str:
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
        
        partition_path = os.path.join(
            base_path,
            f"{timestamp.year:04d}",
            f"{timestamp.month:02d}",
            f"{timestamp.day:02d}"
        )
        return partition_path
    
    def _check_existing_data(self, path: str) -> bool:
        """
        Check if data already exists at the given path.
        
        Args:
            path: Path to check
        
        Returns:
            True if data exists, False otherwise
        """
        path_obj = Path(path)
        if path_obj.exists():
            # Check if directory is not empty
            if path_obj.is_dir() and any(path_obj.iterdir()):
                return True
        return False
    
    def get_dataset_info(self) -> List[Dict[str, str]]:
        """
        Get information about all configured datasets.
        
        Returns:
            List of dictionaries containing dataset information
        """
        dataset_info = []
        for dataset in self.datasets:
            dataset_info.append({
                'name': dataset.get('name'),
                'dataset_id': dataset.get('dataset_id'),
                'description': dataset.get('description', ''),
                'configured_path': dataset.get('download_path')
            })
        return dataset_info
    
    def get_dataset_paths(self) -> Dict[str, str]:
        """
        Get a mapping of dataset names to their actual download paths.
        Useful for downstream processing to know where data is located.
        
        Note: For kagglehub, this returns the kagglehub cache location.
        Call this AFTER fetch() to get actual locations.
        
        Returns:
            Dictionary mapping dataset names to their paths
        """
        paths = {}
        for dataset in self.datasets:
            name = dataset.get('name')
            download_path = dataset.get('download_path')
            
            # Check if data exists at configured path
            if self._check_existing_data(download_path):
                paths[name] = download_path
            else:
                # Data might be in kagglehub cache - note that user should call fetch first
                paths[name] = download_path
        
        return paths


class DataAcquisitionPipeline:
    """Main pipeline for data acquisition."""
    
    def __init__(self, config_path: str):
        """
        Initialize the data acquisition pipeline.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.fetchers = self._initialize_fetchers()
    
    def _load_config(self) -> Dict:
        """
        Load configuration from YAML file.
        
        Returns:
            Configuration dictionary
        """
        logger.info(f"Loading configuration from: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _initialize_fetchers(self) -> Dict[str, DataFetcher]:
        """
        Initialize data fetchers based on configuration.
        
        Returns:
            Dictionary of fetcher instances
        """
        fetchers = {}
        
        data_acquisition_config = self.config.get('data_acquisition', {})
        
        # Initialize Kaggle fetcher if configured
        if 'kaggle' in data_acquisition_config:
            fetchers['kaggle'] = KaggleDataFetcher(
                data_acquisition_config['kaggle']
            )
            logger.info("Initialized Kaggle data fetcher")
        
        return fetchers
    
    def run(self, source: Optional[str] = None, 
            dataset_name: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Run the data acquisition pipeline.
        
        Args:
            source: Optional specific source to fetch from (e.g., 'kaggle')
            dataset_name: Optional specific dataset name to fetch
        
        Returns:
            Dictionary mapping source names to lists of downloaded dataset paths
        """
        logger.info("Starting data acquisition pipeline")
        
        all_paths = {}
        
        # Determine which fetchers to run
        fetchers_to_run = self.fetchers
        if source:
            if source in self.fetchers:
                fetchers_to_run = {source: self.fetchers[source]}
            else:
                logger.error(f"Source '{source}' not found in configuration")
                return {}
        
        # Run each fetcher
        for source_name, fetcher in fetchers_to_run.items():
            logger.info(f"Running {source_name} fetcher")
            try:
                paths = fetcher.fetch(dataset_name=dataset_name)
                all_paths[source_name] = paths
                logger.info(
                    f"Successfully fetched {len(paths)} dataset(s) from {source_name}"
                )
            except Exception as e:
                logger.error(f"Error running {source_name} fetcher: {e}")
                all_paths[source_name] = []
        
        logger.info("Data acquisition pipeline completed")
        return all_paths
    
    def get_data_paths(self) -> Dict[str, Dict[str, str]]:
        """
        Get all dataset paths from configured fetchers.
        
        Returns:
            Dictionary mapping source names to their dataset path mappings
        """
        all_paths = {}
        for source_name, fetcher in self.fetchers.items():
            if hasattr(fetcher, 'get_dataset_paths'):
                all_paths[source_name] = fetcher.get_dataset_paths()
        return all_paths
    
    def get_dataset_info(self) -> Dict[str, List[Dict[str, str]]]:
        """
        Get information about all configured datasets.
        
        Returns:
            Dictionary mapping source names to lists of dataset information
        """
        all_info = {}
        for source_name, fetcher in self.fetchers.items():
            if hasattr(fetcher, 'get_dataset_info'):
                all_info[source_name] = fetcher.get_dataset_info()
        return all_info


def main():
    """Main entry point for the data acquisition script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Fetch medical image datasets from configured sources'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/vision_pipeline.yml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--source',
        type=str,
        choices=['kaggle'],
        help='Specific data source to fetch from'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        help='Specific dataset name to fetch (e.g., lung_cancer_ct_scan)'
    )
    
    args = parser.parse_args()
    
    # Initialize and run pipeline
    pipeline = DataAcquisitionPipeline(args.config)
    pipeline.run(source=args.source, dataset_name=args.dataset)

if __name__ == "__main__":
    main()