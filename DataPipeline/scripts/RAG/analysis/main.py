"""Simple data quality analysis - TFDV style."""

import json
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple, List

import pandas as pd

from validator import DataValidator
from anomalies_and_bias_detection import AnomalyDetector
from drift import DriftDetector

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class DataQualityAnalyzer:
    """Simple data quality analysis pipeline."""

    BASELINE_FILE = "../data/scraped_baseline.jsonl"
    NEW_DATA_FILE = "../data/scraped_updated.jsonl"
    TRAIN_SPLIT_RATIO = 0.7
    BASELINE_STATS_FILE = "baseline_stats.json"

    def __init__(self, is_baseline: bool = True):
        """Initialize analyzer."""
        self.validator = DataValidator()
        self.anomaly_detector = AnomalyDetector()
        self.drift_detector = DriftDetector()
        self.is_baseline = is_baseline

    def load_jsonl(self, filepath: str) -> pd.DataFrame:
        """Load JSONL file."""
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

    def analyze_baseline(self):
        """Analyze baseline data."""
        print("\n" + "="*80)
        print("BASELINE ANALYSIS")
        print("="*80)
        
        # Load data
        baseline_df = self.load_jsonl(self.BASELINE_FILE)
        print(f"Loaded {len(baseline_df)} baseline records")
        
        # Split
        train_df, val_df = self.split_data(baseline_df)
        print(f"Training: {len(train_df)} | Validation: {len(val_df)}")
        print()
        
        # Compute baseline stats from training data
        print("Computing baseline statistics from training data...")
        baseline_stats = self.anomaly_detector.compute_baseline_stats(train_df)
        print(f"Computed stats for {len(baseline_stats)} columns")
        print()
        
        # Schema validation
        print("SCHEMA VALIDATION")
        val_schema = self.validator.validate_schema_completeness(val_df, self._infer_schema(train_df))
        if not val_schema.get('has_issues'):
            print("Validation set schema matches training set")
        else:
            print("Validation set schema issues:")
            if val_schema.get('missing_columns'):
                print(f"  Missing columns: {', '.join(val_schema['missing_columns'])}")
            if val_schema.get('extra_columns'):
                print(f"  Extra columns: {', '.join(val_schema['extra_columns'])}")
            if val_schema.get('type_mismatches'):
                for m in val_schema['type_mismatches']:
                    print(f"  Type mismatch - {m['column']}: expected {m['expected']}, got {m['actual']}")
        print()
        
        # Type validation
        print("DATA TYPE VALIDATION")
        schema = self._infer_schema(train_df)
        val_type_issues = self.validator.validate_data_types(val_df, "Validation", schema)
        if val_type_issues:
            print("Issues found:")
            for col, issue in val_type_issues.items():
                print(f"  {col}: {issue}")
        else:
            print("No type issues detected")
        print()
        
        # GE Validation
        print("VALIDATION CHECKS")
        ge_results = self.validator.validate_with_ge(train_df, val_df, store_baseline=True)
        print(f"Success Rate: {ge_results.get('success_rate', 0):.1f}%")
        print(f"Failed Checks: {ge_results.get('failed_count', 0)}")
        if ge_results.get('failed_expectations'):
            print("\nFailed checks:")
            for exp in ge_results['failed_expectations'][:5]:
                print(f"  - {exp.get('column', 'N/A')}: {exp.get('expectation_type', 'Unknown')}")
                print(f"    Details: {exp.get('details', 'N/A')}")
        print()
        
        # Detect anomalies in validation data
        print("Checking validation data against baseline...")
        val_anomalies = self.anomaly_detector.detect_anomalous_records(
            val_df, baseline_stats, "Validation"
        )
        
        # Detect completeness issues
        val_completeness = self.anomaly_detector.detect_completeness_issues(
            val_df, "Validation"
        )
        
        # Print results
        self._print_anomalies(val_anomalies, "VALIDATION DATA")
        self._print_completeness(val_completeness)
        
        # Detect bias
        print("BIAS ANALYSIS - Training Data")
        train_bias = self.anomaly_detector.detect_bias(train_df)
        self._print_bias(train_bias)
        
        # Save baseline stats
        baseline_data = {
            'stats': baseline_stats,
            'schema': schema,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(self.BASELINE_STATS_FILE, 'w') as f:
            json.dump(baseline_data, f, indent=2, default=str)
        print(f"Baseline stats saved to {self.BASELINE_STATS_FILE}")

    
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
    
    def analyze_new_data(self):
        """Analyze new data against baseline."""
        print("\n" + "="*80)
        print("NEW DATA ANALYSIS")
        print("="*80)
        
        # Load baseline stats
        with open(self.BASELINE_STATS_FILE, 'r') as f:
            baseline_data = json.load(f)
        
        # Handle both old and new format
        if 'stats' in baseline_data:
            baseline_stats = baseline_data['stats']
            schema = baseline_data.get('schema', {})
        else:
            baseline_stats = baseline_data
            schema = {}
        
        print(f"Loaded baseline stats for {len(baseline_stats)} columns")
        
        # Load new data
        new_df = self.load_jsonl(self.NEW_DATA_FILE)
        print(f"Loaded {len(new_df)} new records")
        print()
        
        # Schema validation
        print("SCHEMA VALIDATION")
        schema_val = self.validator.validate_schema_completeness(new_df, schema)
        if not schema_val.get('has_issues'):
            print("New data schema matches baseline")
        else:
            print("Schema issues:")
            if schema_val.get('missing_columns'):
                print(f"  Missing columns: {', '.join(schema_val['missing_columns'])}")
            if schema_val.get('extra_columns'):
                print(f"  Extra columns: {', '.join(schema_val['extra_columns'])}")
            if schema_val.get('type_mismatches'):
                for m in schema_val['type_mismatches']:
                    print(f"  Type mismatch - {m['column']}: expected {m['expected']}, got {m['actual']}")
        print()
        
        # Type validation
        print("DATA TYPE VALIDATION")
        type_issues = self.validator.validate_data_types(new_df, "New Data", schema)
        if type_issues:
            print("Issues found:")
            for col, issue in type_issues.items():
                print(f"  {col}: {issue}")
        else:
            print("No type issues detected")
        print()
        
        # GE Validation
        baseline_df = self.load_jsonl(self.BASELINE_FILE)
        train_df, _ = self.split_data(baseline_df)
        
        print("VALIDATION CHECKS")
        ge_results = self.validator.validate_with_ge(train_df, new_df, store_baseline=False)
        print(f"Success Rate: {ge_results.get('success_rate', 0):.1f}%")
        print(f"Failed Checks: {ge_results.get('failed_count', 0)}")
        if ge_results.get('failed_expectations'):
            print("\nFailed checks:")
            for exp in ge_results['failed_expectations'][:5]:
                print(f"  - {exp.get('column', 'N/A')}: {exp.get('expectation_type', 'Unknown')}")
                print(f"    Details: {exp.get('details', 'N/A')}")
        print()
        
        # Detect anomalies
        print("Checking new data against baseline...")
        anomalies = self.anomaly_detector.detect_anomalous_records(
            new_df, baseline_stats, "New Data"
        )
        
        # Detect completeness issues
        completeness = self.anomaly_detector.detect_completeness_issues(
            new_df, "New Data"
        )
        
        # Print results
        self._print_anomalies(anomalies, "NEW DATA")
        self._print_completeness(completeness)
        
        # Bias analysis
        print("BIAS ANALYSIS - New Data")
        new_bias = self.anomaly_detector.detect_bias(new_df)
        self._print_bias(new_bias)
        
        # Drift detection
        print("DRIFT DETECTION")
        drift_results = self.drift_detector.detect_all_drift(train_df, new_df)
        self._print_drift(drift_results)

    def _print_anomalies(self, anomalies: List[Dict], title: str):
        """Print anomalous records."""
        print(f"BASELINE STATISTICS VIOLATIONS - {title}")
        
        if not anomalies:
            print(" No anomalies detected")
            print()
            return
        
        print(f"Found {len(anomalies)} anomalous records")
        print()
        
        for i, record in enumerate(anomalies[:20], 1):
            print(f"Record {i}:")
            print(f"  Index: {record['index']}")
            
            link = record['link']
            if link != 'N/A':
                print(f"  Link: {link}")
            
            print(f"  Violations ({len(record['violations'])}):")
            for violation in record['violations']:
                print(f"    - {violation}")
            print()
        
        if len(anomalies) > 20:
            print(f"... and {len(anomalies) - 20} more anomalous records")
        print()
    
    def _print_completeness(self, issues: Dict):
        """Print completeness issues."""
        print("DATA COMPLETENESS ISSUES")
        
        if not issues:
            print(" No completeness issues")
            print()
            return
        
        total = sum(len(v) for v in issues.values())
        print(f"Found {total} records with completeness issues")
        print()
        
        for issue_type, records in issues.items():
            print(f"{issue_type}: {len(records)} records")
            for i, record in enumerate(records[:10], 1):
                link = record['link']
                if link != 'N/A':
                    print(f"  {i}. Link: {link}")
                else:
                    print(f"  {i}. Index: {record['index']}")
                print(f"     Reason: {record['reason']}")
            
            if len(records) > 10:
                print(f"  ... and {len(records) - 10} more")
            print()
    
    def _print_bias(self, bias: Dict):
        """Print bias analysis."""
        if not bias:
            print("No bias detected")
            print()
            return
        
        if 'country_bias' in bias:
            print(f"Country Bias: {bias['country_bias']['dominant']} ({bias['country_bias']['percentage']:.1f}%)")
        
        if 'source_bias' in bias:
            print(f"Source Bias: {bias['source_bias']['dominant']} ({bias['source_bias']['percentage']:.1f}%)")
        
        if 'top_topics' in bias:
            print("Top Topics:")
            for topic, count in bias['top_topics'].items():
                print(f"  {topic}: {count}")
        
        if 'temporal_distribution' in bias:
            td = bias['temporal_distribution']
            print(f"Temporal: Last 30d={td['last_30_days']}, 90d={td['last_90_days']}, Year={td['last_year']}")
        
        print()
    
    def _print_drift(self, drift_results: Dict):
        """Print drift analysis with details."""
        if not drift_results:
            print("No drift detected")
            print()
            return
        
        drift_features = [f for f, d in drift_results.items() if d and d.get('has_drift')]
        
        if not drift_features:
            print("No drift detected")
            print()
            return
        
        print(f"Drift detected in {len(drift_features)} features")
        print()
        
        for feature in drift_features[:15]:
            drift_info = drift_results[feature]
            print(f"Feature: {feature}")
            
            if 'mean_shift' in drift_info:
                print(f"  Type: Numeric")
                print(f"  Mean Shift: {drift_info['mean_shift']:.3f} standard deviations")
                print(f"  Baseline Mean: {drift_info['baseline_mean']:.2f}")
                print(f"  New Mean: {drift_info['new_mean']:.2f}")
                if 'percent_mean_change' in drift_info:
                    print(f"  Percent Change: {drift_info['percent_mean_change']:.1f}%")
            else:
                print(f"  Type: Categorical")
                if drift_info.get('new_categories'):
                    print(f"  New Categories: {drift_info['new_categories'][:5]}")
                if drift_info.get('missing_categories'):
                    print(f"  Missing Categories: {drift_info['missing_categories'][:5]}")
            print()
        
        if len(drift_features) > 15:
            print(f"... and {len(drift_features) - 15} more features with drift")
        print()


def main():
    """Run analysis pipeline."""
    print("\n" + "="*80)
    print("DATA QUALITY ANALYSIS PIPELINE")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Phase 1: Baseline
    analyzer = DataQualityAnalyzer(is_baseline=True)
    analyzer.analyze_baseline()
    
    # Phase 2: New Data
    analyzer = DataQualityAnalyzer(is_baseline=False)
    analyzer.analyze_new_data()
    
    print("\n" + "="*80)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


if __name__ == "__main__":
    main()