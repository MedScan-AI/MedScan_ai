"""
Schema and Statistics Generation using Great Expectations and MLMD

This module provides comprehensive data validation and metadata tracking for medical imaging datasets.
It uses Great Expectations for schema inference, statistics generation, anomaly detection,
and validation. It also uses ML Metadata (MLMD) to track all artifacts, executions, and lineage.

Features:
- Generate statistics from CSV data
- Infer and customize schema with domain constraints
- Validate data against schema
- Detect anomalies (missing values, out-of-range, invalid categories)
- Detect drift between baseline and new data
- Generate HTML visualization reports
- Track all artifacts and executions in MLMD
"""

import os
import sys
import yaml
import logging
import pandas as pd
import json
import re
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats as scipy_stats
import numpy as np
from glob import glob
from pathlib import Path

# Great Expectations imports
import great_expectations as gx

# MLflow imports
import mlflow
import mlflow.data
from mlflow.tracking import MlflowClient


class SchemaStatisticsManager:
    """Main class for managing schema, statistics, validation, and metadata tracking."""
    
    def __init__(self, config_path: str):
        """
        Initialize the Schema and Statistics Manager.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._setup_directories()
        self._setup_mlflow()
        self._setup_great_expectations()
        
        self.logger.info("SchemaStatisticsManager initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_file = log_config.get('file', 'logs/schema_statistics.log')
        
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _setup_directories(self):
        """Create necessary directories for outputs."""
        directories = [
            self.config['great_expectations']['statistics']['output_dir'],
            self.config['great_expectations']['statistics']['baseline_dir'],
            self.config['great_expectations']['statistics']['new_data_dir'],
            self.config['great_expectations']['schema']['output_dir'],
            self.config['great_expectations']['validation']['output_dir'],
            self.config['great_expectations']['drift_detection']['output_dir'],
            self.config['great_expectations']['visualization']['output_dir'],
            os.path.dirname(self.config['mlmd']['store']['database_path'])
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            self.logger.debug(f"Created/verified directory: {directory}")
    
    def _get_output_partition_path(self, base_output_dir: str, partition_timestamp: str = None) -> str:
        """
        Generate partitioned output path for ge_outputs.
        
        Args:
            base_output_dir: Base output directory
            partition_timestamp: ISO timestamp from data partition (optional)
            
        Returns:
            Partitioned output path: base_output_dir/YYYY/MM/DD
        """
        if partition_timestamp:
            # Parse timestamp from data partition
            try:
                dt = datetime.fromisoformat(partition_timestamp.replace('Z', '+00:00'))
            except:
                dt = datetime.now()
        else:
            dt = datetime.now()
        
        partition_path = os.path.join(
            base_output_dir,
            f"{dt.year:04d}",
            f"{dt.month:02d}",
            f"{dt.day:02d}"
        )
        os.makedirs(partition_path, exist_ok=True)
        return partition_path
    
    def _setup_mlflow(self):
        """Setup MLflow tracking."""
        # Set MLflow tracking URI
        mlflow_uri = self.config['mlmd']['store']['database_path'].replace('.db', '')
        mlflow_tracking_dir = os.path.join(mlflow_uri, 'mlruns')
        os.makedirs(mlflow_tracking_dir, exist_ok=True)
        
        mlflow.set_tracking_uri(f"file:///{os.path.abspath(mlflow_tracking_dir)}")
        self.mlflow_client = MlflowClient()
        
        # Set experiment name
        experiment_name = "MedScan_Data_Validation"
        try:
            experiment = self.mlflow_client.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = self.mlflow_client.create_experiment(experiment_name)
            else:
                experiment_id = experiment.experiment_id
        except:
            experiment_id = self.mlflow_client.create_experiment(experiment_name)
        
        mlflow.set_experiment(experiment_name)
        self.experiment_id = experiment_id
        self.logger.info(f"MLflow tracking initialized at: {mlflow_tracking_dir}")
        self.logger.info(f"MLflow experiment: {experiment_name} (ID: {experiment_id})")
    
    def _setup_great_expectations(self):
        """Setup Great Expectations context."""
        try:
            # Create a simple in-memory context
            self.ge_context = gx.get_context()
            self.logger.info("Great Expectations context initialized")
        except Exception as e:
            self.logger.warning(f"Could not initialize full GE context: {e}. Using minimal setup.")
            self.ge_context = None
        
        # Load partition metadata
        self._load_partition_metadata()
    
    def _load_partition_metadata(self):
        """Load partition metadata from file."""
        metadata_file = self.config.get('partitioning', {}).get('metadata_file', 'data/partition_metadata.json')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                self.partition_metadata = json.load(f)
            self.logger.info(f"Loaded partition metadata: {len(self.partition_metadata)} entries")
        else:
            self.partition_metadata = {}
            self.logger.info("No existing partition metadata found. Starting fresh.")
    
    def _save_partition_metadata(self):
        """Save partition metadata to file."""
        metadata_file = self.config.get('partitioning', {}).get('metadata_file', 'data/partition_metadata.json')
        os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
        with open(metadata_file, 'w') as f:
            json.dump(self.partition_metadata, f, indent=2)
        self.logger.debug(f"Saved partition metadata to {metadata_file}")
    
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
    
    def _discover_partitions(self, base_path: str) -> List[Dict[str, Any]]:
        """
        Discover all partitions in a base path.
        
        Args:
            base_path: Base directory to search
            
        Returns:
            List of partition info dicts with path and timestamp
        """
        partitions = []
        
        if not os.path.exists(base_path):
            return partitions
        
        # Find all YYYY/MM/DD directories
        pattern = os.path.join(base_path, "[0-9][0-9][0-9][0-9]", "[0-9][0-9]", "[0-9][0-9]")
        partition_dirs = glob(pattern)
        
        for partition_dir in sorted(partition_dirs):
            # Extract year/month/day from path
            parts = Path(partition_dir).parts[-3:]
            try:
                year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                timestamp = datetime(year, month, day)
                
                # Find CSV files in this partition
                csv_files = glob(os.path.join(partition_dir, "*.csv"))
                
                if csv_files:
                    partitions.append({
                        'path': partition_dir,
                        'timestamp': timestamp.isoformat(),
                        'files': csv_files
                    })
            except (ValueError, IndexError) as e:
                self.logger.warning(f"Could not parse partition path {partition_dir}: {e}")
                continue
        
        return partitions
    
    def _load_partition_data(self, partitions: List[Dict[str, Any]]) -> Optional[pd.DataFrame]:
        """
        Load data from partition(s).
        
        Args:
            partitions: List of partition info dicts
            
        Returns:
            Combined DataFrame from all partitions
        """
        if not partitions:
            return None
        
        dfs = []
        for partition in partitions:
            for csv_file in partition['files']:
                try:
                    df = pd.read_csv(csv_file)
                    # Add partition metadata columns
                    df['_partition_timestamp'] = partition['timestamp']
                    df['_partition_path'] = partition['path']
                    dfs.append(df)
                    self.logger.debug(f"Loaded {len(df)} rows from {csv_file}")
                except Exception as e:
                    self.logger.error(f"Error loading {csv_file}: {e}")
        
        if not dfs:
            return None
        
        combined_df = pd.concat(dfs, ignore_index=True)
        self.logger.info(f"Combined {len(dfs)} partition file(s) into {len(combined_df)} total rows")
        return combined_df
    
    def _write_partition_data(self, df: pd.DataFrame, base_path: str, dataset_name: str, 
                             timestamp: datetime = None, is_raw: bool = True) -> str:
        """
        Write data to partitioned directory structure.
        
        Args:
            df: DataFrame to write
            base_path: Base directory path
            dataset_name: Name of the dataset
            timestamp: Optional timestamp for partition
            is_raw: Whether this is raw or preprocessed data
            
        Returns:
            Path to written file
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        partition_path = self._get_partition_path(base_path, timestamp)
        os.makedirs(partition_path, exist_ok=True)
        
        # Remove partition metadata columns before writing
        df_to_write = df.copy()
        for col in ['_partition_timestamp', '_partition_path']:
            if col in df_to_write.columns:
                df_to_write = df_to_write.drop(columns=[col])
        
        file_path = os.path.join(partition_path, f"{dataset_name}.csv")
        df_to_write.to_csv(file_path, index=False)
        self.logger.info(f"Wrote {len(df_to_write)} rows to {file_path}")
        
        # Track in partition metadata
        partition_key = f"{dataset_name}_{timestamp.strftime('%Y%m%d')}"
        if partition_key not in self.partition_metadata:
            self.partition_metadata[partition_key] = {}
        
        data_type = 'raw' if is_raw else 'preprocessed'
        self.partition_metadata[partition_key][data_type] = {
            'path': file_path,
            'timestamp': timestamp.isoformat(),
            'rows': len(df_to_write),
            'columns': len(df_to_write.columns)
        }
        
        self._save_partition_metadata()
        
        return file_path
    
    def _start_mlflow_run(self, run_name: str, tags: Dict[str, str] = None) -> str:
        """
        Start an MLflow run for tracking.
        
        Args:
            run_name: Name of the run
            tags: Optional tags for the run
            
        Returns:
            Run ID
        """
        run = mlflow.start_run(run_name=run_name)
        
        # Set tags
        if tags:
            for key, value in tags.items():
                mlflow.set_tag(key, value)
        
        self.logger.debug(f"Started MLflow run: {run_name} (ID: {run.info.run_id})")
        return run.info.run_id
    
    def _end_mlflow_run(self):
        """End the current MLflow run."""
        mlflow.end_run()
    
    def _log_artifact_file(self, file_path: str, artifact_path: str = None):
        """
        Log a file as an MLflow artifact.
        
        Args:
            file_path: Path to the file to log
            artifact_path: Optional subdirectory in artifact storage
        """
        if os.path.exists(file_path):
            mlflow.log_artifact(file_path, artifact_path)
            self.logger.debug(f"Logged artifact: {file_path}")
    
    def _log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow."""
        mlflow.log_params(params)
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to MLflow."""
        mlflow.log_metrics(metrics)
    
    def _clean_partition_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove partition metadata columns from DataFrame.
        
        Args:
            df: DataFrame potentially containing partition metadata
            
        Returns:
            DataFrame without partition metadata columns
        """
        metadata_cols = ['_partition_timestamp', '_partition_path']
        df_clean = df.copy()
        for col in metadata_cols:
            if col in df_clean.columns:
                df_clean = df_clean.drop(columns=[col])
        return df_clean
    
    def perform_exploratory_analysis(self, dataset_name: str, df: pd.DataFrame, 
                                     output_path: str) -> Dict:
        """
        Perform comprehensive exploratory data analysis on the dataset.
        
        Args:
            dataset_name: Name of the dataset
            df: DataFrame to analyze
            output_path: Path to save EDA report
            
        Returns:
            Dictionary containing EDA results
        """
        self.logger.info(f"Performing exploratory data analysis for {dataset_name}")
        
        eda_results = {
            'dataset_name': dataset_name,
            'timestamp': datetime.now().isoformat(),
            'overview': {},
            'numerical_analysis': {},
            'categorical_analysis': {},
            'missing_data_analysis': {},
            'correlations': {},
            'outliers': {},
            'distributions': {}
        }
        
        # Dataset Overview
        eda_results['overview'] = {
            'num_rows': len(df),
            'num_columns': len(df.columns),
            'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            'duplicate_rows': int(df.duplicated().sum()),
            'column_types': df.dtypes.astype(str).to_dict()
        }
        
        # Missing Data Analysis
        missing_data = {}
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                missing_data[col] = {
                    'count': int(missing_count),
                    'percentage': round(missing_count / len(df) * 100, 2)
                }
        eda_results['missing_data_analysis'] = missing_data
        
        # Numerical Column Analysis
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_cols:
            if col not in ['_partition_timestamp', '_partition_path']:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    eda_results['numerical_analysis'][col] = {
                        'mean': float(col_data.mean()),
                        'median': float(col_data.median()),
                        'std': float(col_data.std()),
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'q25': float(col_data.quantile(0.25)),
                        'q75': float(col_data.quantile(0.75)),
                        'skewness': float(col_data.skew()),
                        'kurtosis': float(col_data.kurtosis())
                    }
                    
                    # Outlier detection using IQR method
                    q1 = col_data.quantile(0.25)
                    q3 = col_data.quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                    
                    eda_results['outliers'][col] = {
                        'count': int(len(outliers)),
                        'percentage': round(len(outliers) / len(col_data) * 100, 2),
                        'lower_bound': float(lower_bound),
                        'upper_bound': float(upper_bound)
                    }
        
        # Categorical Column Analysis
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                value_counts = col_data.value_counts()
                eda_results['categorical_analysis'][col] = {
                    'unique_count': int(col_data.nunique()),
                    'most_common': value_counts.head(10).to_dict(),
                    'mode': str(col_data.mode()[0]) if len(col_data.mode()) > 0 else None,
                    'entropy': float(-sum((value_counts / len(col_data)) * 
                                         np.log2(value_counts / len(col_data))))
                }
        
        # Correlation Analysis (numerical columns only)
        if len(numerical_cols) > 1:
            corr_matrix = df[numerical_cols].corr()
            # Find high correlations
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.5:  # Threshold for "high" correlation
                        high_corr_pairs.append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': float(corr_value)
                        })
            
            eda_results['correlations'] = {
                'high_correlations': high_corr_pairs,
                'correlation_matrix': corr_matrix.to_dict()
            }
        
        # Distribution Analysis
        for col in numerical_cols:
            if col not in ['_partition_timestamp', '_partition_path']:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    # Create histogram data
                    hist, bin_edges = np.histogram(col_data, bins=20)
                    eda_results['distributions'][col] = {
                        'histogram': {
                            'counts': hist.tolist(),
                            'bin_edges': bin_edges.tolist()
                        }
                    }
        
        # Save EDA results
        with open(output_path, 'w') as f:
            json.dump(eda_results, f, indent=2)
        self.logger.info(f"EDA results saved to: {output_path}")
        
        # Log EDA file as artifact
        self._log_artifact_file(output_path, "eda")
        
        # Log key metrics
        self._log_metrics({
            f"{dataset_name}_eda_num_rows": eda_results['overview']['num_rows'],
            f"{dataset_name}_eda_num_columns": eda_results['overview']['num_columns'],
            f"{dataset_name}_eda_duplicate_rows": eda_results['overview']['duplicate_rows'],
            f"{dataset_name}_eda_total_missing": sum(v['count'] for v in missing_data.values()) if missing_data else 0
        })
        
        return eda_results
    
    def generate_eda_html_report(self, dataset_name: str, eda_results: Dict, 
                                 output_path: str) -> None:
        """
        Generate HTML report for EDA results.
        
        Args:
            dataset_name: Name of the dataset
            eda_results: EDA results dictionary
            output_path: Path to save HTML report
        """
        self.logger.info(f"Generating EDA HTML report for {dataset_name}")
        
        overview = eda_results['overview']
        missing = eda_results['missing_data_analysis']
        numerical = eda_results['numerical_analysis']
        categorical = eda_results['categorical_analysis']
        outliers = eda_results['outliers']
        correlations = eda_results.get('correlations', {})
        
        # Build missing data table
        missing_rows = ""
        if missing:
            for col, info in missing.items():
                missing_rows += f"<tr><td>{col}</td><td>{info['count']}</td><td>{info['percentage']}%</td></tr>"
        else:
            missing_rows = "<tr><td colspan='3' style='text-align:center; color:green;'>✓ No missing data detected</td></tr>"
        
        # Build numerical analysis table
        numerical_rows = ""
        for col, stats in numerical.items():
            numerical_rows += f"""
            <tr>
                <td><strong>{col}</strong></td>
                <td>{stats['mean']:.2f}</td>
                <td>{stats['median']:.2f}</td>
                <td>{stats['std']:.2f}</td>
                <td>{stats['min']:.2f}</td>
                <td>{stats['max']:.2f}</td>
                <td>{stats['skewness']:.2f}</td>
            </tr>
            """
        
        # Build categorical analysis table
        categorical_rows = ""
        for col, stats in categorical.items():
            most_common_str = ", ".join([f"{k} ({v})" for k, v in list(stats['most_common'].items())[:5]])
            categorical_rows += f"""
            <tr>
                <td><strong>{col}</strong></td>
                <td>{stats['unique_count']}</td>
                <td>{stats['mode']}</td>
                <td>{most_common_str}</td>
            </tr>
            """
        
        # Build outliers table
        outlier_rows = ""
        for col, info in outliers.items():
            if info['count'] > 0:
                outlier_rows += f"""
                <tr>
                    <td>{col}</td>
                    <td>{info['count']}</td>
                    <td>{info['percentage']}%</td>
                    <td>{info['lower_bound']:.2f} - {info['upper_bound']:.2f}</td>
                </tr>
                """
        if not outlier_rows:
            outlier_rows = "<tr><td colspan='4' style='text-align:center; color:green;'>✓ No outliers detected</td></tr>"
        
        # Build correlations table
        corr_rows = ""
        high_corrs = correlations.get('high_correlations', [])
        if high_corrs:
            for corr in high_corrs[:10]:  # Show top 10
                corr_rows += f"""
                <tr>
                    <td>{corr['feature1']}</td>
                    <td>{corr['feature2']}</td>
                    <td>{corr['correlation']:.3f}</td>
                </tr>
                """
        else:
            corr_rows = "<tr><td colspan='3' style='text-align:center;'>No high correlations found (threshold: 0.5)</td></tr>"
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>EDA Report - {dataset_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
                h2 {{ color: #555; margin-top: 30px; border-bottom: 2px solid #ddd; padding-bottom: 5px; }}
                .container {{ background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .summary-box {{ background-color: #e8f5e9; padding: 15px; border-radius: 5px; margin: 15px 0; }}
                .warning {{ background-color: #fff3cd; padding: 10px; border-left: 4px solid #ffc107; margin: 10px 0; }}
                .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #4CAF50; }}
                .metric-label {{ font-size: 14px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Exploratory Data Analysis Report</h1>
                <p><strong>Dataset:</strong> {dataset_name}</p>
                <p><strong>Generated:</strong> {eda_results['timestamp']}</p>
                
                <div class="summary-box">
                    <h3>Dataset Overview</h3>
                    <div class="metric">
                        <div class="metric-value">{overview['num_rows']:,}</div>
                        <div class="metric-label">Total Rows</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{overview['num_columns']}</div>
                        <div class="metric-label">Total Columns</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{overview['memory_usage_mb']}</div>
                        <div class="metric-label">Memory (MB)</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{overview['duplicate_rows']}</div>
                        <div class="metric-label">Duplicate Rows</div>
                    </div>
                </div>
                
                <h2>Missing Data Analysis</h2>
                <table>
                    <tr><th>Column</th><th>Missing Count</th><th>Missing %</th></tr>
                    {missing_rows}
                </table>
                
                <h2>Numerical Features Analysis</h2>
                <table>
                    <tr>
                        <th>Feature</th><th>Mean</th><th>Median</th><th>Std Dev</th>
                        <th>Min</th><th>Max</th><th>Skewness</th>
                    </tr>
                    {numerical_rows}
                </table>
                
                <h2>Categorical Features Analysis</h2>
                <table>
                    <tr><th>Feature</th><th>Unique Values</th><th>Mode</th><th>Top 5 Values</th></tr>
                    {categorical_rows}
                </table>
                
                <h2>Outlier Detection (IQR Method)</h2>
                <table>
                    <tr><th>Feature</th><th>Outlier Count</th><th>Outlier %</th><th>Valid Range</th></tr>
                    {outlier_rows}
                </table>
                
                <h2>Feature Correlations (|r| > 0.5)</h2>
                <table>
                    <tr><th>Feature 1</th><th>Feature 2</th><th>Correlation</th></tr>
                    {corr_rows}
                </table>
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        self.logger.info(f"EDA HTML report saved to: {output_path}")
        
        # Log HTML report as artifact
        self._log_artifact_file(output_path, "eda")
    
    def _extract_schema_details(self, df: pd.DataFrame) -> Dict:
        """
        Extract detailed schema information from DataFrame.
        
        Args:
            df: DataFrame to extract schema from
            
        Returns:
            Dictionary containing schema details
        """
        schema_details = {
            'columns': {},
            'num_columns': len(df.columns),
            'column_names': list(df.columns),
            'total_rows': len(df)
        }
        
        for col in df.columns:
            dtype = str(df[col].dtype)
            schema_details['columns'][col] = {
                'dtype': dtype,
                'non_null_count': int(df[col].count()),
                'null_count': int(df[col].isnull().sum()),
                'unique_count': int(df[col].nunique())
            }
            
            # Add type-specific details
            if df[col].dtype in ['int64', 'float64']:
                schema_details['columns'][col]['min'] = float(df[col].min()) if not df[col].isnull().all() else None
                schema_details['columns'][col]['max'] = float(df[col].max()) if not df[col].isnull().all() else None
            elif df[col].dtype == 'object':
                schema_details['columns'][col]['sample_values'] = df[col].dropna().head(5).tolist()
        
        return schema_details
    
    def _compare_schemas(self, old_schema: Dict, new_schema: Dict) -> Dict:
        """
        Compare two schemas and detect changes.
        
        Args:
            old_schema: Previous schema details
            new_schema: Current schema details
            
        Returns:
            Dictionary containing schema changes
        """
        changes = {
            'has_changes': False,
            'added_columns': [],
            'removed_columns': [],
            'type_changes': [],
            'summary': ''
        }
        
        old_cols = set(old_schema.get('column_names', []))
        new_cols = set(new_schema.get('column_names', []))
        
        # Detect added columns
        added = new_cols - old_cols
        if added:
            changes['added_columns'] = list(added)
            changes['has_changes'] = True
        
        # Detect removed columns
        removed = old_cols - new_cols
        if removed:
            changes['removed_columns'] = list(removed)
            changes['has_changes'] = True
        
        # Detect type changes
        for col in old_cols & new_cols:
            old_type = old_schema['columns'][col]['dtype']
            new_type = new_schema['columns'][col]['dtype']
            if old_type != new_type:
                changes['type_changes'].append({
                    'column': col,
                    'old_type': old_type,
                    'new_type': new_type
                })
                changes['has_changes'] = True
        
        # Generate summary
        if not changes['has_changes']:
            changes['summary'] = 'No schema changes detected'
        else:
            summary_parts = []
            if changes['added_columns']:
                summary_parts.append(f"{len(changes['added_columns'])} columns added")
            if changes['removed_columns']:
                summary_parts.append(f"{len(changes['removed_columns'])} columns removed")
            if changes['type_changes']:
                summary_parts.append(f"{len(changes['type_changes'])} type changes")
            changes['summary'] = ', '.join(summary_parts)
        
        return changes
    
    def _load_previous_schema(self, schema_output_dir: str, dataset_name: str) -> Optional[Dict]:
        """
        Load the most recent previous schema if it exists.
        Searches current directory and parent partitions for previous schemas.
        
        Args:
            schema_output_dir: Directory where schemas are stored (may be partitioned)
            dataset_name: Name of the dataset
            
        Returns:
            Previous schema details or None if not found
        """
        # Check if we're in a partitioned directory (e.g., .../2025/10/23)
        schema_base_dir = schema_output_dir
        if re.match(r'.*[\\/]\d{4}[\\/]\d{2}[\\/]\d{2}$', schema_output_dir):
            # We're in a partition, search parent directory for all partitions
            schema_base_dir = os.path.dirname(os.path.dirname(os.path.dirname(schema_output_dir)))
        
        # Search for all schema files recursively
        schema_files = glob(os.path.join(schema_base_dir, "**", f"{dataset_name}_schema.json"), recursive=True)
        
        if not schema_files:
            return None
        
        # Get most recent schema file by modification time
        schema_files.sort(key=os.path.getmtime, reverse=True)
        
        # Skip the first one if it's the current file we're about to write
        for schema_file in schema_files:
            # Skip if this is the exact current directory (file we're about to create)
            if os.path.dirname(schema_file) == schema_output_dir:
                continue
            
            try:
                with open(schema_file, 'r') as f:
                    schema = json.load(f)
                self.logger.info(f"Loaded previous schema from: {schema_file}")
                return schema.get('schema_details')
            except Exception as e:
                self.logger.warning(f"Could not load schema from {schema_file}: {e}")
                continue
        
        return None
    
    def generate_statistics(self, dataset_name: str, df: pd.DataFrame, 
                          output_path: str) -> Dict:
        """
        Generate statistics for a dataset using Great Expectations.
        
        Args:
            dataset_name: Name of the dataset
            df: Pandas DataFrame
            output_path: Path to save statistics
            
        Returns:
            Statistics dictionary
        """
        self.logger.info(f"Generating statistics for {dataset_name}")
        
        # Log dataset info as parameters
        self._log_params({
            f"{dataset_name}_num_rows": len(df),
            f"{dataset_name}_num_columns": len(df.columns)
        })
        
        # Generate statistics using pandas describe
        stats = {
            'num_rows': len(df),
            'num_columns': len(df.columns),
            'column_names': list(df.columns),
            'numerical_stats': {},
            'categorical_stats': {}
        }
        
        # Numerical statistics
        for col in df.select_dtypes(include=[np.number]).columns:
            stats['numerical_stats'][col] = {
                'count': int(df[col].count()),
                'mean': float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                'std': float(df[col].std()) if not pd.isna(df[col].std()) else None,
                'min': float(df[col].min()) if not pd.isna(df[col].min()) else None,
                'max': float(df[col].max()) if not pd.isna(df[col].max()) else None,
                'missing': int(df[col].isna().sum())
            }
        
        # Categorical statistics
        for col in df.select_dtypes(include=['object', 'category']).columns:
            stats['categorical_stats'][col] = {
                'count': int(df[col].count()),
                'unique': int(df[col].nunique()),
                'missing': int(df[col].isna().sum()),
                'top_values': df[col].value_counts().head(10).to_dict()
            }
        
        # Save statistics
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        self.logger.info(f"Statistics saved to: {output_path}")
        
        # Log statistics file as artifact
        self._log_artifact_file(output_path, "statistics")
        
        # Log metrics
        self._log_metrics({
            f"{dataset_name}_missing_values": sum(
                s.get('missing', 0) for s in stats['numerical_stats'].values()
            ) + sum(
                s.get('missing', 0) for s in stats['categorical_stats'].values()
            )
        })
        
        return stats
    
    def infer_schema(self, dataset_name: str, df: pd.DataFrame, 
                    output_path: str) -> Dict:
        """
        Infer schema and create expectations using Great Expectations.
        Also extracts detailed schema information and detects schema changes.
        
        Args:
            dataset_name: Name of the dataset
            df: DataFrame for schema inference
            output_path: Path to save schema
            
        Returns:
            Expectations dictionary
        """
        self.logger.info(f"Inferring schema for {dataset_name}")
        
        # Extract detailed schema information
        current_schema_details = self._extract_schema_details(df)
        self.logger.info(f"Schema details extracted: {current_schema_details['num_columns']} columns, {current_schema_details['total_rows']} rows")
        
        # Load previous schema if exists
        schema_output_dir = os.path.dirname(output_path)
        previous_schema_details = self._load_previous_schema(schema_output_dir, dataset_name)
        
        # Compare schemas if previous exists
        schema_changes = None
        if previous_schema_details:
            schema_changes = self._compare_schemas(previous_schema_details, current_schema_details)
            self.logger.info(f"Schema comparison: {schema_changes['summary']}")
            
            if schema_changes['has_changes']:
                self.logger.warning(f"⚠️  Schema changes detected!")
                if schema_changes['added_columns']:
                    self.logger.warning(f"  Added columns: {', '.join(schema_changes['added_columns'])}")
                if schema_changes['removed_columns']:
                    self.logger.warning(f"  Removed columns: {', '.join(schema_changes['removed_columns'])}")
                if schema_changes['type_changes']:
                    for change in schema_changes['type_changes']:
                        self.logger.warning(f"  Type change in '{change['column']}': {change['old_type']} → {change['new_type']}")
        else:
            self.logger.info("No previous schema found. This is the baseline schema.")
        
        # Create Great Expectations suite
        expectations = {
            'dataset_name': dataset_name,
            'expectations': [],
            'schema_details': current_schema_details,
            'schema_changes': schema_changes,
            'timestamp': datetime.now().isoformat()
        }
        
        # Apply domain constraints from config
        constraints = self.config['schema_constraints']
        
        # Numerical constraints
        for col, config in constraints['numerical_features'].items():
            if col in df.columns:
                expectations['expectations'].append({
                    'expectation_type': 'expect_column_values_to_be_between',
                    'column': col,
                    'min_value': config['min'],
                    'max_value': config['max'],
                    'required': config.get('required', False)
                })
                if config.get('required', False):
                    expectations['expectations'].append({
                        'expectation_type': 'expect_column_values_to_not_be_null',
                        'column': col
                    })
        
        # Categorical constraints
        for col, config in constraints['categorical_features'].items():
            if col in df.columns:
                if 'allowed_values' in config:
                    expectations['expectations'].append({
                        'expectation_type': 'expect_column_values_to_be_in_set',
                        'column': col,
                        'value_set': config['allowed_values']
                    })
                if config.get('required', False):
                    expectations['expectations'].append({
                        'expectation_type': 'expect_column_values_to_not_be_null',
                        'column': col
                    })
        
        # String constraints
        for col, config in constraints['string_features'].items():
            if col in df.columns:
                if config.get('required', False):
                    expectations['expectations'].append({
                        'expectation_type': 'expect_column_values_to_not_be_null',
                        'column': col
                    })
                if config.get('unique', False):
                    expectations['expectations'].append({
                        'expectation_type': 'expect_column_values_to_be_unique',
                        'column': col
                    })
        
        # Save schema (partition directory already provides temporal context)
        with open(output_path, 'w') as f:
            json.dump(expectations, f, indent=2)
        self.logger.info(f"Schema saved to: {output_path}")
        
        # Log schema file as artifact
        self._log_artifact_file(output_path, "schemas")
        
        # Log parameters
        self._log_params({
            f"{dataset_name}_num_expectations": len(expectations['expectations']),
            f"{dataset_name}_num_columns": current_schema_details['num_columns'],
            f"{dataset_name}_schema_has_changes": schema_changes['has_changes'] if schema_changes else False
        })
        
        # Log schema change metrics
        if schema_changes:
            self._log_metrics({
                f"{dataset_name}_schema_added_columns": len(schema_changes['added_columns']),
                f"{dataset_name}_schema_removed_columns": len(schema_changes['removed_columns']),
                f"{dataset_name}_schema_type_changes": len(schema_changes['type_changes'])
            })
        
        return expectations
    
    def validate_data(self, dataset_name: str, df: pd.DataFrame, expectations: Dict,
                     output_path: str) -> Dict:
        """
        Validate data against expectations and detect anomalies.
        
        Args:
            dataset_name: Name of the dataset
            df: DataFrame to validate
            expectations: Expectations/schema dict
            output_path: Path to save validation results
            
        Returns:
            Validation results dictionary
        """
        self.logger.info(f"Validating data for {dataset_name}")
        
        anomalies = []
        
        # Validate each expectation
        for expectation in expectations['expectations']:
            exp_type = expectation['expectation_type']
            col = expectation.get('column')
            
            if exp_type == 'expect_column_values_to_be_between':
                min_val = expectation['min_value']
                max_val = expectation['max_value']
                violations = df[(df[col] < min_val) | (df[col] > max_val)]
                if len(violations) > 0:
                    anomalies.append({
                        'column': col,
                        'expectation': exp_type,
                        'description': f"{len(violations)} values out of range [{min_val}, {max_val}]"
                    })
            
            elif exp_type == 'expect_column_values_to_be_in_set':
                allowed = expectation['value_set']
                violations = df[~df[col].isin(allowed)]
                if len(violations) > 0:
                    anomalies.append({
                        'column': col,
                        'expectation': exp_type,
                        'description': f"{len(violations)} values not in allowed set"
                    })
            
            elif exp_type == 'expect_column_values_to_not_be_null':
                violations = df[df[col].isna()]
                if len(violations) > 0:
                    anomalies.append({
                        'column': col,
                        'expectation': exp_type,
                        'description': f"{len(violations)} null values found"
                    })
            
            elif exp_type == 'expect_column_values_to_be_unique':
                duplicates = df[df[col].duplicated()]
                if len(duplicates) > 0:
                    anomalies.append({
                        'column': col,
                        'expectation': exp_type,
                        'description': f"{len(duplicates)} duplicate values found"
                    })
        
        validation_results = {
            'dataset_name': dataset_name,
            'is_valid': len(anomalies) == 0,
            'num_anomalies': len(anomalies),
            'anomalies': anomalies,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save validation results
        with open(output_path, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        # Log anomalies
        if len(anomalies) > 0:
            self.logger.warning(f"Found {len(anomalies)} anomalies in {dataset_name}")
            for anomaly in anomalies:
                self.logger.warning(f"  - {anomaly['column']}: {anomaly['description']}")
        else:
            self.logger.info(f"No anomalies found in {dataset_name}")
        
        # Log validation file as artifact
        self._log_artifact_file(output_path, "validations")
        
        # Log metrics
        self._log_metrics({
            f"{dataset_name}_is_valid": 1.0 if len(anomalies) == 0 else 0.0,
            f"{dataset_name}_num_anomalies": float(len(anomalies))
        })
        
        return validation_results
    
    def detect_drift(self, dataset_name: str, baseline_df: pd.DataFrame, new_df: pd.DataFrame,
                    output_path: str) -> Dict:
        """
        Detect drift between baseline and new data using statistical tests.
        
        Args:
            dataset_name: Name of the dataset
            baseline_df: Baseline DataFrame
            new_df: New data DataFrame
            output_path: Path to save drift report
            
        Returns:
            Drift report dictionary
        """
        self.logger.info(f"Detecting drift for {dataset_name}")
        
        drift_config = self.config['great_expectations']['drift_detection']
        drifted_features = []
        
        # Detect drift for numerical features
        for col in baseline_df.select_dtypes(include=[np.number]).columns:
            if col in new_df.columns:
                # Kolmogorov-Smirnov test
                statistic, pvalue = scipy_stats.ks_2samp(
                    baseline_df[col].dropna(),
                    new_df[col].dropna()
                )
                
                # Check against threshold
                threshold = drift_config['statistical_test_threshold']
                if pvalue < threshold:
                    drifted_features.append({
                        'feature': col,
                        'type': 'numerical',
                        'test': 'Kolmogorov-Smirnov',
                        'statistic': float(statistic),
                        'pvalue': float(pvalue),
                        'drift_detected': True
                    })
        
        # Detect drift for categorical features
        for col in baseline_df.select_dtypes(include=['object', 'category']).columns:
            if col in new_df.columns:
                # Chi-square test
                baseline_counts = baseline_df[col].value_counts()
                new_counts = new_df[col].value_counts()
                
                # Align categories
                all_categories = set(baseline_counts.index) | set(new_counts.index)
                baseline_freq = [baseline_counts.get(cat, 0) for cat in all_categories]
                new_freq = [new_counts.get(cat, 0) for cat in all_categories]
                
                try:
                    statistic, pvalue = scipy_stats.chisquare(new_freq, baseline_freq)
                    
                    threshold = drift_config['statistical_test_threshold']
                    if pvalue < threshold:
                        drifted_features.append({
                            'feature': col,
                            'type': 'categorical',
                            'test': 'Chi-square',
                            'statistic': float(statistic),
                            'pvalue': float(pvalue),
                            'drift_detected': True
                        })
                except Exception as e:
                    self.logger.warning(f"Could not perform chi-square test for {col}: {e}")
        
        drift_report = {
            'dataset_name': dataset_name,
            'baseline_size': len(baseline_df),
            'new_data_size': len(new_df),
            'has_drift': len(drifted_features) > 0,
            'num_drifted_features': len(drifted_features),
            'drifted_features': drifted_features,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save drift report
        with open(output_path, 'w') as f:
            json.dump(drift_report, f, indent=2)
        
        # Log drift
        if len(drifted_features) > 0:
            self.logger.warning(f"Drift detected in {dataset_name}")
            for feature in drifted_features:
                self.logger.warning(f"  - {feature['feature']} ({feature['type']}): p-value={feature['pvalue']:.4f}")
        else:
            self.logger.info(f"No drift detected in {dataset_name}")
        
        # Log drift report as artifact
        self._log_artifact_file(output_path, "drift")
        
        # Log metrics
        self._log_metrics({
            f"{dataset_name}_has_drift": 1.0 if len(drifted_features) > 0 else 0.0,
            f"{dataset_name}_num_drifted_features": float(len(drifted_features))
        })
        
        return drift_report
    
    def generate_reports(self, dataset_name: str, baseline_stats: Dict, new_stats: Dict,
                        expectations: Dict, validation_results: Dict, drift_report: Dict = None,
                        partition_timestamp: str = None):
        """
        Generate HTML visualization reports.
        
        Args:
            dataset_name: Name of the dataset
            baseline_stats: Baseline statistics
            new_stats: New data statistics
            expectations: Schema expectations
            validation_results: Validation results
            drift_report: Drift report (optional)
            partition_timestamp: Timestamp for partitioned output (optional)
        """
        self.logger.info(f"Generating visualization reports for {dataset_name}")
        
        viz_config = self.config['great_expectations']['visualization']
        base_output_dir = viz_config['output_dir']
        
        # Use partitioned output path
        output_dir = self._get_output_partition_path(base_output_dir, partition_timestamp)
        
        # Generate statistics comparison HTML
        stats_html = self._generate_statistics_html(baseline_stats, new_stats)
        stats_report_path = os.path.join(output_dir, f"{dataset_name}_statistics_report.html")
        with open(stats_report_path, 'w', encoding='utf-8') as f:
            f.write(stats_html)
        self.logger.info(f"Statistics report saved to: {stats_report_path}")
        
        # Generate validation report HTML
        validation_html = self._generate_validation_html(validation_results)
        validation_report_path = os.path.join(output_dir, f"{dataset_name}_validation_report.html")
        with open(validation_report_path, 'w', encoding='utf-8') as f:
            f.write(validation_html)
        self.logger.info(f"Validation report saved to: {validation_report_path}")
        
        # Generate drift report HTML
        if drift_report:
            drift_html = self._generate_drift_html(drift_report)
            drift_report_path = os.path.join(output_dir, f"{dataset_name}_drift_report.html")
            with open(drift_report_path, 'w', encoding='utf-8') as f:
                f.write(drift_html)
            self.logger.info(f"Drift report saved to: {drift_report_path}")
            # Log drift report as artifact
            self._log_artifact_file(drift_report_path, "reports")
        
        # Log HTML reports as artifacts
        self._log_artifact_file(stats_report_path, "reports")
        self._log_artifact_file(validation_report_path, "reports")
    
    def _generate_statistics_html(self, baseline_stats: Dict, new_stats: Dict) -> str:
        """Generate HTML for statistics comparison."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Statistics Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .section {{ margin: 30px 0; }}
            </style>
        </head>
        <body>
            <h1>Statistics Comparison Report</h1>
            <div class="section">
                <h2>Dataset Overview</h2>
                <table>
                    <tr><th>Metric</th><th>Baseline</th><th>New Data</th></tr>
                    <tr><td>Number of Rows</td><td>{}</td><td>{}</td></tr>
                    <tr><td>Number of Columns</td><td>{}</td><td>{}</td></tr>
                </table>
            </div>
            <div class="section">
                <h2>Numerical Features</h2>
                <p>Baseline features: {}</p>
                <p>New data features: {}</p>
            </div>
            <div class="section">
                <h2>Categorical Features</h2>
                <p>Baseline features: {}</p>
                <p>New data features: {}</p>
            </div>
        </body>
        </html>
        """.format(
            baseline_stats['num_rows'],
            new_stats['num_rows'],
            baseline_stats['num_columns'],
            new_stats['num_columns'],
            ', '.join(baseline_stats['numerical_stats'].keys()),
            ', '.join(new_stats['numerical_stats'].keys()),
            ', '.join(baseline_stats['categorical_stats'].keys()),
            ', '.join(new_stats['categorical_stats'].keys())
        )
        return html
    
    def _generate_validation_html(self, validation_results: Dict) -> str:
        """Generate HTML for validation results."""
        anomalies_rows = ""
        for anomaly in validation_results['anomalies']:
            anomalies_rows += f"<tr><td>{anomaly['column']}</td><td>{anomaly['expectation']}</td><td>{anomaly['description']}</td></tr>"
        
        if not anomalies_rows:
            anomalies_rows = "<tr><td colspan='3' style='text-align:center; color:green;'>✓ No anomalies detected</td></tr>"
        
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .valid {{ color: green; font-weight: bold; }}
                .invalid {{ color: red; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>Data Validation Report</h1>
            <h2>Summary</h2>
            <p>Dataset: {}</p>
            <p>Status: <span class="{}">{}</span></p>
            <p>Number of Anomalies: {}</p>
            <p>Timestamp: {}</p>
            
            <h2>Anomalies</h2>
            <table>
                <tr><th>Column</th><th>Expectation</th><th>Description</th></tr>
                {}
            </table>
        </body>
        </html>
        """.format(
            validation_results['dataset_name'],
            'valid' if validation_results['is_valid'] else 'invalid',
            'VALID' if validation_results['is_valid'] else 'INVALID',
            validation_results['num_anomalies'],
            validation_results['timestamp'],
            anomalies_rows
        )
        return html
    
    def _generate_drift_html(self, drift_report: Dict) -> str:
        """Generate HTML for drift report."""
        drift_rows = ""
        for feature in drift_report['drifted_features']:
            drift_rows += f"<tr><td>{feature['feature']}</td><td>{feature['type']}</td><td>{feature['test']}</td><td>{feature['pvalue']:.6f}</td></tr>"
        
        if not drift_rows:
            drift_rows = "<tr><td colspan='4' style='text-align:center; color:green;'>✓ No drift detected</td></tr>"
        
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Drift Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .drift {{ color: orange; font-weight: bold; }}
                .no-drift {{ color: green; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>Data Drift Detection Report</h1>
            <h2>Summary</h2>
            <p>Dataset: {}</p>
            <p>Baseline Size: {} rows</p>
            <p>New Data Size: {} rows</p>
            <p>Drift Status: <span class="{}">{}</span></p>
            <p>Drifted Features: {}</p>
            <p>Timestamp: {}</p>
            
            <h2>Drifted Features</h2>
            <table>
                <tr><th>Feature</th><th>Type</th><th>Statistical Test</th><th>P-Value</th></tr>
                {}
            </table>
        </body>
        </html>
        """.format(
            drift_report['dataset_name'],
            drift_report['baseline_size'],
            drift_report['new_data_size'],
            'drift' if drift_report['has_drift'] else 'no-drift',
            'DRIFT DETECTED' if drift_report['has_drift'] else 'NO DRIFT',
            drift_report['num_drifted_features'],
            drift_report['timestamp'],
            drift_rows
        )
        return html
    
    def process_dataset(self, dataset_key: str):
        """
        Process a single dataset through the complete pipeline with partition support.
        
        Args:
            dataset_key: Key of the dataset in configuration
        """
        self.logger.info(f"=" * 80)
        self.logger.info(f"Processing dataset: {dataset_key}")
        self.logger.info(f"=" * 80)
        
        # Get dataset configuration
        dataset_config = self.config['datasets'][dataset_key]
        dataset_name = dataset_config['name']
        raw_base_path = dataset_config.get('raw_base_path', dataset_config.get('path'))
        preprocessed_base_path = dataset_config.get('preprocessed_base_path')
        
        # Use metadata_base_path if configured, otherwise fall back to preprocessed_base_path
        metadata_base_path = dataset_config.get('metadata_base_path')
        metadata_filename = dataset_config.get('metadata_filename')
        
        # Determine the primary data source path
        if metadata_base_path:
            data_source_path = metadata_base_path
            data_source_label = "metadata"
            self.logger.info(f"Using metadata source: {metadata_base_path}")
        else:
            data_source_path = preprocessed_base_path
            data_source_label = "preprocessed"
            self.logger.info(f"Using preprocessed data source: {preprocessed_base_path}")
        
        # Check if partitioning is enabled
        partitioning_enabled = self.config.get('partitioning', {}).get('enabled', False)
        
        # Start MLflow run for this dataset
        run_id = self._start_mlflow_run(
            run_name=f"{dataset_name}_validation",
            tags={
                "dataset": dataset_name,
                "dataset_key": dataset_key,
                "pipeline": "data_validation",
                "partitioning_enabled": str(partitioning_enabled),
                "data_source": data_source_label
            }
        )
        
        try:
            # Discover and load partitions
            if partitioning_enabled:
                # Discover existing data partitions
                data_partitions = self._discover_partitions(data_source_path)
                self.logger.info(f"Found {len(data_partitions)} {data_source_label} partition(s)")
                
                # Load data
                if not data_partitions:
                    self.logger.warning(f"No data partitions found for {dataset_key}. Skipping.")
                    return
                
                # Determine baseline and new data based on partitions
                baseline_df = None
                new_df = None
                baseline_timestamp = None
                new_timestamp = None
                
                if len(data_partitions) >= 2:
                    # Use two most recent partitions for drift detection
                    baseline_partition = [data_partitions[-2]]  # Second most recent
                    new_partition = [data_partitions[-1]]  # Most recent
                    
                    baseline_df = self._load_partition_data(baseline_partition)
                    new_df = self._load_partition_data(new_partition)
                    baseline_timestamp = baseline_partition[0]['timestamp']
                    new_timestamp = new_partition[0]['timestamp']
                    
                    self.logger.info(f"Using partition-based drift detection (2 most recent):")
                    self.logger.info(f"  Baseline: {baseline_timestamp} ({len(baseline_df)} rows)")
                    self.logger.info(f"  New: {new_timestamp} ({len(new_df)} rows)")
                    
                    if len(data_partitions) > 2:
                        self.logger.info(f"  Note: Found {len(data_partitions)} total partitions, comparing latest 2")
                else:
                    # Only one partition - use it as baseline, no drift detection
                    baseline_df = self._load_partition_data(data_partitions)
                    new_df = None
                    baseline_timestamp = data_partitions[0]['timestamp']
                    new_timestamp = None
                    self.logger.info(f"Only 1 partition found. Drift detection will be skipped.")
                    self.logger.info(f"  Baseline: {baseline_timestamp} ({len(baseline_df)} rows)")
            else:
                # Non-partitioned mode: load from single CSV
                if metadata_base_path and metadata_filename:
                    # Use metadata path with filename
                    dataset_path = os.path.join(metadata_base_path, metadata_filename)
                else:
                    # Fall back to old 'path' config
                    dataset_path = dataset_config.get('path')
                
                if not dataset_path or not os.path.exists(dataset_path):
                    self.logger.error(f"Dataset path not found: {dataset_path}")
                    return
                
                self.logger.info(f"Loading dataset from: {dataset_path}")
                baseline_df = pd.read_csv(dataset_path)
                new_df = None
                baseline_timestamp = None  # No partition timestamp in non-partitioned mode
                new_timestamp = None
                self.logger.info(f"Loaded {len(baseline_df)} rows and {len(baseline_df.columns)} columns")
                data_partitions = []  # Initialize for logging
            
            # Log dataset parameters
            self._log_params({
                "dataset_name": dataset_name,
                "partitioning_enabled": partitioning_enabled,
                "num_partitions": len(data_partitions) if partitioning_enabled else 1,
                "baseline_rows": len(baseline_df) if baseline_df is not None else 0,
                "new_data_rows": len(new_df) if new_df is not None else 0
            })
            
            # Get execution configuration
            exec_config = self.config['execution']
            operations = exec_config['operations']
            
            # Clean partition metadata columns from DataFrames
            baseline_df_clean = self._clean_partition_metadata(baseline_df) if baseline_df is not None else None
            new_df_clean = self._clean_partition_metadata(new_df) if new_df is not None else None
            
            # Generate statistics
            baseline_stats = None
            new_stats = None
            
            if operations['generate_statistics']:
                # Use partitioned output paths
                stats_base_dir = self.config['great_expectations']['statistics']['baseline_dir']
                stats_output_dir = self._get_output_partition_path(stats_base_dir, baseline_timestamp)
                stats_output_path = os.path.join(stats_output_dir, f"{dataset_name}_stats.json")
                
                baseline_stats = self.generate_statistics(
                    dataset_name=f"{dataset_name}_baseline",
                    df=baseline_df_clean,
                    output_path=stats_output_path
                )
                
                # Generate statistics for new data only if it exists
                if new_df_clean is not None:
                    new_stats_base_dir = self.config['great_expectations']['statistics']['new_data_dir']
                    new_stats_output_dir = self._get_output_partition_path(new_stats_base_dir, new_timestamp)
                    new_stats_output_path = os.path.join(new_stats_output_dir, f"{dataset_name}_new_stats.json")
                    
                    new_stats = self.generate_statistics(
                        dataset_name=f"{dataset_name}_new",
                        df=new_df_clean,
                        output_path=new_stats_output_path
                    )
            
            # Infer schema
            expectations = None
            if operations['infer_schema']:
                # Use partitioned output path (use new timestamp if available, otherwise baseline)
                schema_timestamp = new_timestamp if new_timestamp else baseline_timestamp
                schema_base_dir = self.config['great_expectations']['schema']['output_dir']
                schema_output_dir = self._get_output_partition_path(schema_base_dir, schema_timestamp)
                schema_output_path = os.path.join(schema_output_dir, f"{dataset_name}_schema.json")
                
                expectations = self.infer_schema(
                    dataset_name=dataset_name,
                    df=baseline_df_clean,
                    output_path=schema_output_path
                )
            
            # Perform EDA on latest partition
            eda_results = None
            if operations.get('perform_eda', True):
                # Use new data if available, otherwise use baseline
                eda_df = new_df_clean if new_df_clean is not None else baseline_df_clean
                eda_partition_label = "latest" if new_df_clean is not None else "baseline"
                eda_timestamp = new_timestamp if new_timestamp else baseline_timestamp
                
                # Use partitioned output path
                eda_base_dir = self.config.get('eda', {}).get('output_dir', 'data/ge_outputs/eda')
                eda_output_dir = self._get_output_partition_path(eda_base_dir, eda_timestamp)
                
                eda_output_path = os.path.join(eda_output_dir, f"{dataset_name}_eda_{eda_partition_label}.json")
                eda_results = self.perform_exploratory_analysis(
                    dataset_name=f"{dataset_name}_{eda_partition_label}",
                    df=eda_df,
                    output_path=eda_output_path
                )
                
                # Generate HTML report for EDA
                if operations.get('generate_reports', True):
                    eda_html_path = os.path.join(eda_output_dir, f"{dataset_name}_eda_{eda_partition_label}.html")
                    self.generate_eda_html_report(
                        dataset_name=f"{dataset_name}_{eda_partition_label}",
                        eda_results=eda_results,
                        output_path=eda_html_path
                    )
            
            # Validate data
            validation_results = None
            if operations['validate_data'] and expectations:
                # Use partitioned output path
                validation_timestamp = new_timestamp if new_timestamp else baseline_timestamp
                validation_base_dir = self.config['great_expectations']['validation']['output_dir']
                validation_output_dir = self._get_output_partition_path(validation_base_dir, validation_timestamp)
                validation_output_path = os.path.join(validation_output_dir, f"{dataset_name}_validation.json")
                
                validation_results = self.validate_data(
                    dataset_name=dataset_name,
                    df=baseline_df_clean,
                    expectations=expectations,
                    output_path=validation_output_path
                )
            
            # Detect drift - only if new data exists
            drift_report = None
            if operations['detect_drift'] and new_df_clean is not None and new_stats:
                # Use partitioned output path (use new timestamp for drift reports)
                drift_base_dir = self.config['great_expectations']['drift_detection']['output_dir']
                drift_output_dir = self._get_output_partition_path(drift_base_dir, new_timestamp)
                drift_output_path = os.path.join(drift_output_dir, f"{dataset_name}_drift.json")
                
                drift_report = self.detect_drift(
                    dataset_name=dataset_name,
                    baseline_df=baseline_df_clean,
                    new_df=new_df_clean,
                    output_path=drift_output_path
                )
            elif operations['detect_drift'] and new_df_clean is None:
                self.logger.info("Drift detection skipped: insufficient partitions (need at least 2)")
            
            # Generate reports
            if operations['generate_reports']:
                # Use new timestamp if available, otherwise baseline timestamp
                report_timestamp = new_timestamp if new_timestamp else baseline_timestamp
                self.generate_reports(
                    dataset_name=dataset_name,
                    baseline_stats=baseline_stats,
                    new_stats=new_stats if new_stats else baseline_stats,
                    expectations=expectations,
                    validation_results=validation_results,
                    drift_report=drift_report,
                    partition_timestamp=report_timestamp
                )
            
            self.logger.info(f"Completed processing for {dataset_key}")
            
        finally:
            # End MLflow run
            self._end_mlflow_run()
    
    def run_pipeline(self):
        """Run the complete data validation pipeline for all configured datasets."""
        self.logger.info("Starting Data Validation Pipeline")
        self.logger.info(f"Timestamp: {datetime.now().isoformat()}")
        
        # Get datasets to process
        datasets_to_process = self.config['execution']['process_datasets']
        
        # Process each dataset
        for dataset_key in datasets_to_process:
            try:
                self.process_dataset(dataset_key)
            except Exception as e:
                self.logger.error(f"Error processing dataset {dataset_key}: {e}", exc_info=True)
        
        self.logger.info("=" * 80)
        self.logger.info("Data Validation Pipeline completed successfully")
        self.logger.info("=" * 80)
    
    def print_summary(self):
        """Print a summary of all MLflow tracked runs and artifacts."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("MLFLOW TRACKING SUMMARY")
        self.logger.info("=" * 80)
        
        # Get all runs from the experiment
        runs = self.mlflow_client.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=["start_time DESC"]
        )
        
        self.logger.info(f"\nTotal Runs: {len(runs)}")
        self.logger.info(f"Experiment ID: {self.experiment_id}")
        self.logger.info(f"Tracking URI: {mlflow.get_tracking_uri()}")
        
        # Display recent runs
        for run in runs[:10]:  # Show last 10 runs
            self.logger.info(f"\n  Run: {run.info.run_name}")
            self.logger.info(f"    Run ID: {run.info.run_id}")
            self.logger.info(f"    Status: {run.info.status}")
            self.logger.info(f"    Start Time: {run.info.start_time}")
            
            # Show key metrics
            if run.data.metrics:
                self.logger.info(f"    Metrics: {len(run.data.metrics)} recorded")
                for key, value in list(run.data.metrics.items())[:3]:
                    self.logger.info(f"      {key}: {value}")
            
            # Show artifacts
            try:
                artifacts = self.mlflow_client.list_artifacts(run.info.run_id)
                if artifacts:
                    self.logger.info(f"    Artifacts: {len(artifacts)} files")
            except:
                pass
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("To view the MLflow UI, run:")
        self.logger.info(f"  mlflow ui --backend-store-uri {mlflow.get_tracking_uri()}")
        self.logger.info("=" * 80)


def main():
    """
    Main entry point for automated Great Expectations + MLflow data validation pipeline.
    
    This function:
    1. Loads configuration from metadata.yml
    2. Initializes MLflow tracking and Great Expectations components
    3. Processes all configured datasets:
       - Generates statistics
       - Infers and applies schema with domain constraints
       - Validates data and detects anomalies
       - Detects drift between baseline and new data
       - Generates HTML visualization reports
    4. Tracks all artifacts, metrics, and parameters in MLflow
    5. Prints summary of tracked runs
    
    Usage:
        python schema_statistics.py --config config/metadata.yml
    
    Configuration:
        All parameters are in the specified config file
    
    MLflow UI:
        After running, view results with: mlflow ui
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run data validation pipeline with Great Expectations and MLflow'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/metadata.yml',
        help='Path to configuration file (default: config/metadata.yml)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("Great Expectations + MLflow Pipeline - MedScan AI")
    print("=" * 80)
    
    # Ensure we're in the Data-Pipeline directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_pipeline_dir = os.path.dirname(os.path.dirname(script_dir))
    
    # Change to Data-Pipeline directory if not already there
    if os.path.basename(os.getcwd()) != 'Data-Pipeline':
        if os.path.exists(os.path.join(data_pipeline_dir, 'config', 'metadata.yml')):
            os.chdir(data_pipeline_dir)
            print(f"Changed working directory to: {data_pipeline_dir}")
    
    # Get config path
    config_path = args.config
    
    try:
        # Initialize manager with automatic setup
        print(f"\nInitializing from config: {config_path}")
        manager = SchemaStatisticsManager(config_path)
        
        # Run automated pipeline for all datasets
        print("\nStarting automated data validation pipeline...")
        manager.run_pipeline()
        
        # Print MLflow tracking summary
        print("\nGenerating MLflow tracking summary...")
        manager.print_summary()
        
        print("\n" + "=" * 80)
        print("Pipeline completed successfully!")
        print("=" * 80)
        print("\nOutputs generated in data/ directory:")
        print("  ✓ Statistics: data/ge_outputs/baseline/ and data/ge_outputs/new_data/")
        print("  ✓ Schemas: data/ge_outputs/schemas/")
        print("  ✓ Validation reports: data/ge_outputs/validations/")
        print("  ✓ Drift detection: data/ge_outputs/drift/")
        print("  ✓ HTML visualizations: data/ge_outputs/reports/")
        print("  ✓ MLflow tracking: data/mlflow_store/mlruns/")
        print("  ✓ Logs: data/logs/schema_statistics.log")
        print("\nView results in MLflow UI:")
        print("  cd Data-Pipeline && mlflow ui --backend-store-uri file:///$(pwd)/data/mlflow_store/mlruns")
        print("  Then open: http://localhost:5000")
        print()
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: Configuration or data file not found - {e}")
        print("  Please ensure config/metadata.yml and data files exist.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error during pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
