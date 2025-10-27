"""Data quality analysis - TFDV style."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import pandas as pd

from .validator import DataValidator
from .anomalies_and_bias_detection import AnomalyDetector
from .drift import DriftDetector

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class DataQualityAnalyzer:
    """Simple data quality analysis pipeline."""

    # GCS paths (source of truth)
    GCS_BASELINE_PATH = "RAG/raw_data/baseline/baseline.jsonl"
    GCS_BASELINE_STATS_PATH = "RAG/validation/baseline_stats.json"
    
    # Local working directory
    WORK_DIR = Path("/opt/airflow/data/validation_work")
    
    # Local temporary files
    LOCAL_BASELINE = WORK_DIR / "baseline.jsonl"
    LOCAL_NEW_DATA = WORK_DIR / "new_data.jsonl"
    LOCAL_STATS = WORK_DIR / "baseline_stats.json"
    
    TRAIN_SPLIT_RATIO = 0.7

    def __init__(self, gcs_manager=None, is_baseline: bool = True):
        """Initialize analyzer."""
        self.validator = DataValidator()
        self.anomaly_detector = AnomalyDetector()
        self.drift_detector = DriftDetector()
        self.is_baseline = is_baseline
        self.gcs = gcs_manager
        
        # Ensure working directory exists
        self.WORK_DIR.mkdir(parents=True, exist_ok=True)

    def load_jsonl(self, filepath) -> pd.DataFrame:
        """Load JSONL file."""
        # Convert to Path if string
        filepath = Path(filepath) if isinstance(filepath, str) else filepath
        
        records = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    records.append(json.loads(line.strip()))
                except:
                    continue
        
        df = pd.DataFrame(records)
        
        # Remove error records
        if "error" in df.columns:
            df = df[df["error"].isna()].reset_index(drop=True)
        
        # Parse dates
        for col in df.columns:
            if df[col].dtype == 'object':
                parsed = pd.to_datetime(df[col], errors='coerce')
                if parsed.notna().sum() > len(df) * 0.5:
                    df[col] = parsed
        
        return df

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split into train/validation."""
        split_idx = int(len(df) * self.TRAIN_SPLIT_RATIO)
        return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

    def generate_baseline_stats(self, baseline_df: pd.DataFrame) -> Dict:
        """Generate and save baseline statistics."""
        
        train_df, val_df = self.split_data(baseline_df)
        print(f"Training: {len(train_df)} | Validation: {len(val_df)}")
        
        # Compute stats from training data
        baseline_stats = self.anomaly_detector.compute_baseline_stats(train_df)
        schema = self._infer_schema(train_df)
        
        # Run validation checks on validation split
        print("\nVALIDATION CHECKS (on validation split)")
        ge_results = self.validator.validate_with_ge(train_df, val_df, store_baseline=True)
        print(f"Success Rate: {ge_results.get('success_rate', 0):.1f}%")
        
        # Package results
        baseline_data = {
            'stats': baseline_stats,
            'schema': schema,
            'validation_results': ge_results,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'train_size': len(train_df),
            'val_size': len(val_df)
        }
        
        # Save locally
        with open(self.LOCAL_STATS, 'w') as f:
            json.dump(baseline_data, f, indent=2, default=str)
        
        print(f"Stats saved to {self.LOCAL_STATS}")
        
        # Upload to GCS if manager available
        if self.gcs:
            self.gcs.upload_file(str(self.LOCAL_STATS), self.GCS_BASELINE_STATS_PATH)
            
        return baseline_data

    def validate_against_baseline(self, new_df: pd.DataFrame) -> Dict:
        """Validate new data against baseline."""
        # Download baseline stats if not present
        if not self.LOCAL_STATS.exists() and self.gcs:
            success = self.gcs.download_file(
                self.GCS_BASELINE_STATS_PATH, 
                str(self.LOCAL_STATS)
            )
            if not success:
                raise FileNotFoundError(
                    f"Baseline stats not found in GCS: {self.GCS_BASELINE_STATS_PATH}"
                )
        
        if not self.LOCAL_STATS.exists():
            raise FileNotFoundError(
                "Baseline stats not found. Run baseline generation first."
            )
        
        # Load baseline stats
        with open(self.LOCAL_STATS, 'r') as f:
            baseline_data = json.load(f)
        
        baseline_stats = baseline_data['stats']
        schema = baseline_data['schema']
        
        # Schema validation
        schema_val = self.validator.validate_schema_completeness(new_df, schema)
        if not schema_val.get('has_issues'):
            print("Schema matches baseline")
        else:
            print("Schema issues detected:")
            if schema_val.get('missing_columns'):
                print(f"  Missing: {', '.join(schema_val['missing_columns'])}")
            if schema_val.get('extra_columns'):
                print(f"  Extra: {', '.join(schema_val['extra_columns'])}")
        print()
        
        # Type validation
        print("DATA TYPE VALIDATION")
        type_issues = self.validator.validate_data_types(new_df, "New Data", schema)
        if type_issues:
            print("Type issues found:")
            for col, issue in type_issues.items():
                print(f"  {col}: {issue}")
        else:
            print("No type issues")
        print()
        
        # Anomaly detection
        print("ANOMALY DETECTION")
        anomalies = self.anomaly_detector.detect_anomalous_records(
            new_df, baseline_stats, "New Data"
        )
        print(f"Found {len(anomalies)} anomalous records")
        print()
        
        # Completeness check
        print("COMPLETENESS CHECK")
        completeness = self.anomaly_detector.detect_completeness_issues(
            new_df, "New Data"
        )
        total_issues = sum(len(v) for v in completeness.values())
        print(f"Found {total_issues} completeness issues")
        print()
        
        # Bias analysis
        print("BIAS ANALYSIS")
        bias = self.anomaly_detector.detect_bias(new_df)
        self._print_bias(bias)
        
        # Drift detection (need baseline data)
        print("DRIFT DETECTION")
        if self.gcs and self.gcs.blob_exists(self.GCS_BASELINE_PATH):
            self.gcs.download_file(self.GCS_BASELINE_PATH, str(self.LOCAL_BASELINE))
            baseline_df = self.load_jsonl(self.LOCAL_BASELINE)
            train_df, _ = self.split_data(baseline_df)
            
            drift_results = self.drift_detector.detect_all_drift(train_df, new_df)
            drift_features = [f for f, d in drift_results.items() if d and d.get('has_drift')]
            print(f"Drift detected in {len(drift_features)} features")
        else:
            print("Baseline data not available for drift detection")
            drift_results = {}
        print()
        
        # Compile results
        results = {
            'schema_validation': schema_val,
            'type_validation': type_issues,
            'anomalies': {
                'records': anomalies,
                'total': len(anomalies),
                'percentage': len(anomalies) / len(new_df) * 100 if len(new_df) > 0 else 0
            },
            'completeness': completeness,
            'bias': bias,
            'drift': drift_results,
            'timestamp': datetime.now().isoformat(),
            'total_records': len(new_df)
        }
        
        return results

    def _infer_schema(self, df: pd.DataFrame) -> Dict:
        """Infer schema from DataFrame."""
        schema = {}
        for col in df.columns:
            dtype = df[col].dtype
            if pd.api.types.is_integer_dtype(dtype):
                schema[col] = {"type": "int"}
            elif pd.api.types.is_float_dtype(dtype):
                schema[col] = {"type": "float"}
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                schema[col] = {"type": "datetime"}
            elif pd.api.types.is_object_dtype(dtype):
                sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                if isinstance(sample, list):
                    schema[col] = {"type": "list"}
                else:
                    schema[col] = {"type": "str"}
        return schema
    
    def _print_bias(self, bias: Dict):
        """Print bias analysis."""
        if not bias:
            print("No bias detected")
            return
        
        if 'country_bias' in bias:
            print(f"Country: {bias['country_bias']['dominant']} ({bias['country_bias']['percentage']:.1f}%)")
        
        if 'source_bias' in bias:
            print(f"Source: {bias['source_bias']['dominant']} ({bias['source_bias']['percentage']:.1f}%)")
        
        if 'top_topics' in bias:
            print("Top Topics:")
            for topic, count in list(bias['top_topics'].items())[:3]:
                print(f"  {topic}: {count}")
        print()


def main():
    """Run analysis pipeline - for standalone testing only."""
    print("\n" + "="*80)
    print("DATA QUALITY ANALYSIS PIPELINE")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # This is only for local testing
    BASELINE_FILE = Path("/opt/airflow/data/scraped_baseline.jsonl")
    NEW_DATA_FILE = Path("/opt/airflow/data/scraped_updated.jsonl")
    
    # Phase 1: Baseline
    analyzer = DataQualityAnalyzer(is_baseline=True)
    baseline_df = analyzer.load_jsonl(BASELINE_FILE)
    analyzer.generate_baseline_stats(baseline_df)
    
    # Phase 2: New Data
    analyzer = DataQualityAnalyzer(is_baseline=False)
    new_df = analyzer.load_jsonl(NEW_DATA_FILE)
    results = analyzer.validate_against_baseline(new_df)
    
    print("\n" + "="*80)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


if __name__ == "__main__":
    main()