"""
model_loader.py - Load latest trained models for TB and Lung Cancer detection
Supports both local filesystem and GCS (Google Cloud Storage)
Models are located in: gs://bucket/vision/trained_models/{BUILD_ID}/models/models/YYYY/MM/DD/timestamp/
"""
import os
import json
import logging
from pathlib import Path
from typing import Optional, Tuple, List
import tensorflow as tf

# Try to import GCS client (optional)
try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if not GCS_AVAILABLE:
    logger.warning("Google Cloud Storage not available. GCS model loading disabled.")


class ModelLoader:
    """Load latest trained ResNet models for inference."""
    
    def __init__(self, models_base_path: str = None, gcs_bucket: str = None, gcs_prefix: str = None):
        """
        Initialize ModelLoader.
        
        Args:
            models_base_path: Base path to models directory (local filesystem).
                             Defaults to ModelDevelopment/data/models
            gcs_bucket: GCS bucket name (if using GCS for models)
            gcs_prefix: GCS prefix/path to models directory
                       Default: "vision/trained_models" (will find latest BUILD_ID)
        """
        self.gcs_bucket = gcs_bucket or os.getenv("GCS_BUCKET_NAME", "medscan-pipeline-medscanai-476500")
        self.gcs_prefix = gcs_prefix or os.getenv("GCS_MODELS_PREFIX", "vision/trained_models")
        self.use_gcs = (self.gcs_bucket is not None and GCS_AVAILABLE) or os.getenv("USE_GCS_MODELS", "false").lower() == "true"
        
        if models_base_path is None:
            # Try to find the models directory relative to this file
            script_dir = Path(__file__).parent
            # Go up to ModelDevelopment, then to data/models
            possible_paths = [
                script_dir.parent.parent / "data" / "models",  # If VisionInference is at ModelDevelopment/VisionInference
                script_dir.parent / "data" / "models",  # Alternative structure
                Path("/app/data/models"),  # Docker path
                Path("data/models"),  # Relative path
            ]
            
            for path in possible_paths:
                if path.exists():
                    models_base_path = str(path)
                    break
            else:
                # Use default relative to script
                models_base_path = str(script_dir.parent.parent / "data" / "models")
        
        self.models_base_path = Path(models_base_path)
        
        if self.use_gcs:
            logger.info(f"Using GCS for models: gs://{self.gcs_bucket}/{self.gcs_prefix}")
            self.storage_client = storage.Client()
            self.bucket = self.storage_client.bucket(self.gcs_bucket)
        else:
            logger.info(f"Using local filesystem for models: {self.models_base_path}")
            self.storage_client = None
            self.bucket = None
        
        self.tb_model = None
        self.lung_cancer_model = None
        self.tb_model_path = None
        self.lung_cancer_model_path = None
        self.tb_class_names = None
        self.lung_cancer_class_names = None
    
    def _find_latest_build_id(self) -> Optional[str]:
        """
        Find the latest BUILD_ID in vision/trained_models/.
        
        Returns:
            Latest BUILD_ID or None
        """
        if not self.use_gcs:
            return None
        
        try:
            # List all BUILD_IDs in vision/trained_models/
            prefix = f"{self.gcs_prefix}/"
            blobs = self.bucket.list_blobs(prefix=prefix, delimiter='/')
            
            build_ids = []
            for page in blobs.pages:
                for prefix in page.prefixes:
                    # Extract BUILD_ID from prefix (e.g., "vision/trained_models/BUILD_ID/")
                    build_id = prefix.rstrip('/').split('/')[-1]
                    if build_id:
                        build_ids.append(build_id)
            
            if not build_ids:
                logger.warning("No BUILD_IDs found in vision/trained_models/")
                return None
            
            # Sort and return latest (assuming UUIDs or timestamps)
            # For UUIDs, we'll use the most recent one found
            latest_build_id = sorted(build_ids)[-1]
            logger.info(f"Found latest BUILD_ID: {latest_build_id}")
            return latest_build_id
            
        except Exception as e:
            logger.error(f"Error finding latest BUILD_ID: {e}")
            return None
    
    def _find_latest_model_gcs(self, dataset_name: str) -> Optional[str]:
        """
        Find the latest model in GCS for a given dataset.
        Searches: gs://bucket/vision/trained_models/{BUILD_ID}/models/models/YYYY/MM/DD/timestamp/{dataset}_CNN_ResNet18/
        
        Args:
            dataset_name: Dataset name ('tb' or 'lung_cancer_ct_scan')
            
        Returns:
            GCS URI to the latest model file or None
        """
        if not self.use_gcs:
            return None
        
        # Find latest BUILD_ID
        build_id = self._find_latest_build_id()
        if not build_id:
            logger.warning(f"No BUILD_ID found in {self.gcs_prefix}/")
            return None
        
        # Search path: vision/trained_models/{BUILD_ID}/models/models/YYYY/MM/DD/timestamp/{dataset}_CNN_ResNet18/
        base_prefix = f"{self.gcs_prefix}/{build_id}/models/models/"
        logger.info(f"Searching for {dataset_name} model in: {base_prefix}")
        
        try:
            # List all year directories
            all_models = []
            current_build_id = build_id  # Store for use in model info
            
            # List blobs with delimiter to get "directories"
            blobs = self.bucket.list_blobs(prefix=base_prefix, delimiter='/')
            
            # Get year directories
            year_dirs = []
            for page in blobs.pages:
                for prefix in page.prefixes:
                    year = prefix.replace(base_prefix, '').rstrip('/')
                    if year.isdigit():
                        year_dirs.append((int(year), prefix))
            
            if not year_dirs:
                logger.warning(f"No year directories found in {base_prefix}")
                return None
            
            # Sort by year (descending)
            year_dirs.sort(reverse=True)
            
            # Search through years, months, days, timestamps
            for year, year_prefix in year_dirs:
                # Get month directories
                month_blobs = self.bucket.list_blobs(prefix=year_prefix, delimiter='/')
                month_dirs = []
                for page in month_blobs.pages:
                    for prefix in page.prefixes:
                        month = prefix.replace(year_prefix, '').rstrip('/')
                        if month.isdigit():
                            month_dirs.append((int(month), prefix))
                
                month_dirs.sort(reverse=True)
                
                for month, month_prefix in month_dirs:
                    # Get day directories
                    day_blobs = self.bucket.list_blobs(prefix=month_prefix, delimiter='/')
                    day_dirs = []
                    for page in day_blobs.pages:
                        for prefix in page.prefixes:
                            day = prefix.replace(month_prefix, '').rstrip('/')
                            if day.isdigit():
                                day_dirs.append((int(day), prefix))
                    
                    day_dirs.sort(reverse=True)
                    
                    for day, day_prefix in day_dirs:
                        # Get timestamp directories
                        timestamp_blobs = self.bucket.list_blobs(prefix=day_prefix, delimiter='/')
                        timestamp_dirs = []
                        for page in timestamp_blobs.pages:
                            for prefix in page.prefixes:
                                timestamp = prefix.replace(day_prefix, '').rstrip('/')
                                if timestamp:
                                    timestamp_dirs.append((timestamp, prefix))
                        
                        # Sort timestamps (treat as strings for lexicographic sort, or convert to int if numeric)
                        timestamp_dirs.sort(key=lambda x: (int(x[0]) if x[0].isdigit() else 0, x[0]), reverse=True)
                        
                        for timestamp, timestamp_prefix in timestamp_dirs:
                            # Look for model directory
                            model_dir_name = f"{dataset_name}_CNN_ResNet18"
                            model_prefix = f"{timestamp_prefix}{model_dir_name}/"
                            
                            # List model files
                            model_blobs = list(self.bucket.list_blobs(prefix=model_prefix))
                            model_files = [b.name for b in model_blobs if b.name.endswith('.keras') or b.name.endswith('.h5')]
                            
                            if model_files:
                                # Prefer _final.keras or _best.keras
                                best_model = None
                                for model_file in model_files:
                                    if '_final' in model_file or '_best' in model_file:
                                        best_model = model_file
                                        break
                                
                                if best_model is None:
                                    best_model = model_files[0]
                                
                                # Convert timestamp to int for comparison if it's numeric
                                # Timestamps are typically HHMMSS format (6 digits)
                                try:
                                    if timestamp.isdigit():
                                        timestamp_int = int(timestamp)
                                    else:
                                        # If not numeric, use 0 (will be sorted last)
                                        timestamp_int = 0
                                except:
                                    timestamp_int = 0
                                
                                # Store with datetime info for comparison
                                all_models.append({
                                    'gcs_path': f"gs://{self.gcs_bucket}/{best_model}",
                                    'datetime': (year, month, day, timestamp_int),
                                    'timestamp_str': timestamp,
                                    'build_id': current_build_id  # Store build_id for reference
                                })
                                logger.info(f"Found model: {best_model} (date: {year}/{month:02d}/{day:02d}, time: {timestamp})")
            
            # Select latest by datetime
            if all_models:
                all_models.sort(key=lambda x: x['datetime'], reverse=True)
                latest = all_models[0]
                logger.info(f"Selected latest model: {latest['gcs_path']}")
                return latest['gcs_path']
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding model in GCS: {e}", exc_info=True)
            return None
    
    def _find_latest_model_local(self, dataset_name: str) -> Optional[Path]:
        """
        Find the latest model for a given dataset (local filesystem).
        
        Args:
            dataset_name: Dataset name ('tb' or 'lung_cancer_ct_scan')
            
        Returns:
            Path to the latest model file or None if not found
        """
        if not self.models_base_path.exists():
            logger.error(f"Models base path does not exist: {self.models_base_path}")
            return None
        
        # Search for models in YYYY/MM/DD/timestamp/{dataset_name}_CNN_ResNet18/
        all_models = []
        
        # Iterate through year directories
        for year_dir in sorted(self.models_base_path.iterdir(), reverse=True):
            if not year_dir.is_dir() or not year_dir.name.isdigit():
                continue
            
            year = int(year_dir.name)
            
            # Iterate through month directories
            for month_dir in sorted(year_dir.iterdir(), reverse=True):
                if not month_dir.is_dir() or not month_dir.name.isdigit():
                    continue
                
                month = int(month_dir.name)
                
                # Iterate through day directories
                for day_dir in sorted(month_dir.iterdir(), reverse=True):
                    if not day_dir.is_dir() or not day_dir.name.isdigit():
                        continue
                    
                    day = int(day_dir.name)
                    
                    # Iterate through timestamp directories
                    for timestamp_dir in sorted(day_dir.iterdir(), reverse=True):
                        if not timestamp_dir.is_dir():
                            continue
                        
                        # Look for model directory matching dataset
                        model_dir_name = f"{dataset_name}_CNN_ResNet18"
                        model_dir = timestamp_dir / model_dir_name
                        
                        if model_dir.exists() and model_dir.is_dir():
                            # Look for model files
                            model_files = list(model_dir.glob("*.keras"))
                            if not model_files:
                                # Try .h5 format
                                model_files = list(model_dir.glob("*.h5"))
                            
                            if model_files:
                                # Prefer _best.keras or _final.keras, otherwise take first
                                best_model = None
                                for model_file in model_files:
                                    if "_best" in model_file.name or "_final" in model_file.name:
                                        best_model = model_file
                                        break
                                
                                if best_model is None:
                                    best_model = model_files[0]
                                
                                # Parse timestamp (HHMMSS format)
                                timestamp_str = timestamp_dir.name
                                try:
                                    timestamp_int = int(timestamp_str) if timestamp_str.isdigit() else 0
                                except ValueError:
                                    timestamp_int = 0
                                
                                # Store model with date-time tuple for comparison
                                all_models.append({
                                    'path': best_model,
                                    'datetime': (year, month, day, timestamp_int),
                                    'timestamp_str': timestamp_str
                                })
                                logger.info(f"Found model: {best_model} (date: {year}/{month:02d}/{day:02d}, time: {timestamp_str})")
        
        # Select the latest model by comparing date-time tuples
        if all_models:
            # Sort by datetime tuple (year, month, day, timestamp) in descending order
            all_models.sort(key=lambda x: x['datetime'], reverse=True)
            latest_model_path = all_models[0]['path']
            latest_datetime = all_models[0]['datetime']
            logger.info(f"Selected latest model: {latest_model_path} (date: {latest_datetime[0]}/{latest_datetime[1]:02d}/{latest_datetime[2]:02d}, time: {all_models[0]['timestamp_str']})")
            return latest_model_path
        else:
            return None
    
    def _download_model_from_gcs(self, gcs_path: str, local_path: Path) -> bool:
        """
        Download model from GCS to local path.
        
        Args:
            gcs_path: GCS URI (gs://bucket/path/to/model.keras)
            local_path: Local path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Parse GCS path
            if not gcs_path.startswith('gs://'):
                logger.error(f"Invalid GCS path: {gcs_path}")
                return False
            
            path_parts = gcs_path.replace('gs://', '').split('/', 1)
            bucket_name = path_parts[0]
            blob_path = path_parts[1] if len(path_parts) > 1 else ''
            
            # Download blob
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            
            # Create parent directory
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download
            logger.info(f"Downloading model from {gcs_path} to {local_path}")
            blob.download_to_filename(str(local_path))
            logger.info(f"Model downloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading model from GCS: {e}", exc_info=True)
            return False
    
    def _find_latest_model(self, dataset_name: str) -> Optional[Path]:
        """
        Find the latest model for a given dataset (supports both local and GCS).
        For GCS: Downloads model to local temp directory for loading.
        
        Args:
            dataset_name: Dataset name ('tb' or 'lung_cancer_ct_scan')
            
        Returns:
            Path to the latest model file or None if not found
        """
        if self.use_gcs:
            # Find model in GCS
            gcs_path = self._find_latest_model_gcs(dataset_name)
            if gcs_path:
                # Download to temporary location for TensorFlow to load
                import tempfile
                temp_dir = Path(tempfile.mkdtemp(prefix="models_"))
                model_filename = Path(gcs_path).name
                local_path = temp_dir / model_filename
                
                if self._download_model_from_gcs(gcs_path, local_path):
                    logger.info(f"Model downloaded from GCS to: {local_path}")
                    # Store GCS path for metadata loading
                    self._last_gcs_model_path = gcs_path
                    return local_path
                else:
                    logger.error(f"Failed to download model from GCS: {gcs_path}")
                    return None
            else:
                logger.warning(f"Model not found in GCS for {dataset_name}, trying local filesystem")
                return self._find_latest_model_local(dataset_name)
        else:
            # Use local filesystem
            return self._find_latest_model_local(dataset_name)
    
    def _load_metadata(self, model_path: Path) -> Optional[dict]:
        """
        Load training metadata for a model.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Metadata dictionary or None
        """
        try:
            # If model was downloaded from GCS, try to find metadata in GCS
            if self.use_gcs:
                # Try to find metadata in the same GCS location as the model
                # Extract path components from the original GCS path
                # We need to search for training_metadata.json in the timestamp directory
                # For now, we'll try to find it by searching GCS
                try:
                    # Get the model's GCS path info (stored during download)
                    if hasattr(self, '_last_gcs_model_path'):
                        gcs_model_path = self._last_gcs_model_path
                        # Construct metadata path: same directory as model, but training_metadata.json
                        # Path structure: .../timestamp/{dataset}_CNN_ResNet18/model.keras
                        # Metadata: .../timestamp/training_metadata.json
                        gcs_metadata_path = '/'.join(gcs_model_path.split('/')[:-2]) + '/training_metadata.json'
                        
                        # Download metadata from GCS
                        blob = self.bucket.blob(gcs_metadata_path.replace(f'gs://{self.gcs_bucket}/', ''))
                        if blob.exists():
                            metadata_content = blob.download_as_text()
                            metadata = json.loads(metadata_content)
                            logger.info(f"Loaded metadata from GCS: {gcs_metadata_path}")
                            return metadata
                except Exception as e:
                    logger.debug(f"Could not load metadata from GCS: {e}")
            
            # Try local filesystem
            timestamp_dir = model_path.parent.parent  # Go up from model_dir to timestamp
            metadata_file = timestamp_dir / "training_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"Loaded metadata from: {metadata_file}")
                return metadata
            
            logger.debug(f"Metadata file not found")
            return None
        except Exception as e:
            logger.warning(f"Error loading metadata: {e}")
            return None
    
    def load_tb_model(self) -> bool:
        """
        Load the latest TB model.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            model_path = self._find_latest_model("tb")
            if model_path is None:
                logger.error("TB model not found")
                return False
            
            logger.info(f"Loading TB model from: {model_path}")
            self.tb_model = tf.keras.models.load_model(str(model_path))
            self.tb_model_path = model_path
            
            # Try to load class names from metadata
            metadata = self._load_metadata(model_path)
            if metadata and 'class_names' in metadata:
                self.tb_class_names = metadata['class_names']
                logger.info(f"Loaded TB class names from metadata: {self.tb_class_names}")
            else:
                # Default class names (update based on your training data)
                self.tb_class_names = ["Normal", "Tuberculosis"]
                logger.info(f"Using default TB class names: {self.tb_class_names}")
            
            logger.info("TB model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading TB model: {e}", exc_info=True)
            return False
    
    def load_lung_cancer_model(self) -> bool:
        """
        Load the latest Lung Cancer model.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            model_path = self._find_latest_model("lung_cancer_ct_scan")
            if model_path is None:
                logger.error("Lung Cancer model not found")
                return False
            
            logger.info(f"Loading Lung Cancer model from: {model_path}")
            self.lung_cancer_model = tf.keras.models.load_model(str(model_path))
            self.lung_cancer_model_path = model_path
            
            # Try to load class names from metadata
            metadata = self._load_metadata(model_path)
            if metadata and 'class_names' in metadata:
                self.lung_cancer_class_names = metadata['class_names']
                logger.info(f"Loaded Lung Cancer class names from metadata: {self.lung_cancer_class_names}")
            else:
                # Infer class names from model output shape
                num_classes = self.lung_cancer_model.output_shape[-1]
                if num_classes == 2:
                    self.lung_cancer_class_names = ["Normal", "Lung_Cancer"]
                elif num_classes == 3:
                    self.lung_cancer_class_names = ["Adenocarcinoma", "Large_Cell_Carcinoma", "Squamous_Cell_Carcinoma"]
                elif num_classes == 4:
                    self.lung_cancer_class_names = ["Adenocarcinoma", "Large_Cell_Carcinoma", "Normal", "Squamous_Cell_Carcinoma"]
                elif num_classes == 6:
                    # 6-class lung cancer dataset (alphabetically sorted)
                    self.lung_cancer_class_names = [
                        "Adenocarcinoma", 
                        "Benign",
                        "Large_Cell_Carcinoma", 
                        "Malignant",
                        "Normal", 
                        "Squamous_Cell_Carcinoma"
                    ]
                else:
                    # Generic class names
                    self.lung_cancer_class_names = [f"Class_{i}" for i in range(num_classes)]
                logger.warning(f"No metadata found. Using inferred class names ({num_classes} classes): {self.lung_cancer_class_names}")
                logger.warning(f"For accurate class names, ensure training_metadata.json exists in GCS alongside the model.")
            
            logger.info("Lung Cancer model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading Lung Cancer model: {e}", exc_info=True)
            return False
    
    def load_all_models(self) -> Tuple[bool, bool]:
        """
        Load all models.
        
        Returns:
            Tuple of (tb_loaded, lung_cancer_loaded)
        """
        tb_loaded = self.load_tb_model()
        lung_cancer_loaded = self.load_lung_cancer_model()
        return tb_loaded, lung_cancer_loaded
    
    def get_tb_model(self):
        """Get the loaded TB model."""
        return self.tb_model
    
    def get_lung_cancer_model(self):
        """Get the loaded Lung Cancer model."""
        return self.lung_cancer_model
    
    def get_tb_class_names(self) -> List[str]:
        """Get class names for TB model."""
        return self.tb_class_names or ["Normal", "Tuberculosis"]
    
    def get_lung_cancer_class_names(self) -> List[str]:
        """Get class names for Lung Cancer model."""
        return self.lung_cancer_class_names or ["Normal", "Lung_Cancer"]
