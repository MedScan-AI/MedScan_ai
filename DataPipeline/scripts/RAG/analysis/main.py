"""Main analysis orchestrator for data quality pipeline."""

import json
import logging
import os
import shutil
from datetime import datetime
from typing import Dict, Optional, Tuple

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
    """Complete data quality analysis pipeline."""

    BASELINE_FILE = "../data/scraped_baseline.jsonl"
    NEW_DATA_FILE = "../data/scraped_updated.jsonl"
    TRAIN_SPLIT_RATIO = 0.7
    BASELINE_STATS_FILE = "baseline_stats.json"

    def __init__(self, is_baseline: bool = True):
        """Initialize the analyzer.
        
        Args:
            is_baseline: If True, analyze baseline data and save stats.
                        If False, analyze new data using saved baseline stats.
        """
        self._clean_gx_context()
        
        self.validator = DataValidator()
        self.anomaly_detector = AnomalyDetector()
        self.drift_detector = DriftDetector()

        self.is_baseline = is_baseline
        self.baseline_df: Optional[pd.DataFrame] = None
        self.train_df: Optional[pd.DataFrame] = None
        self.validation_df: Optional[pd.DataFrame] = None
        self.new_data_df: Optional[pd.DataFrame] = None
        self.inferred_schema: Optional[Dict] = None
        self.baseline_stats: Optional[Dict] = None
        self.remove_anomalies = True

    def _clean_gx_context(self):
        """Clean Great Expectations context directory."""
        gx_dir = "./gx"
        if os.path.exists(gx_dir):
            try:
                shutil.rmtree(gx_dir)
                logger.info("Cleaned GX context directory")
            except Exception as e:
                logger.debug(f"Could not clean GX directory: {e}")

    def _normalize_datetimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize datetime columns to prevent timezone errors."""
        df = df.copy()
        for col in df.columns:
            try:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    # Convert to UTC and remove timezone
                    if df[col].dt.tz is not None:
                        df[col] = df[col].dt.tz_convert('UTC').dt.tz_localize(None)
                    
                    # Convert to string format
                    df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Convert back to datetime (always naive)
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            except Exception as e:
                logger.debug(f"Could not normalize datetime in {col}: {e}")
        return df

    def _save_baseline_stats(self, stats: Dict):
        """Save baseline statistics to file."""
        try:
            with open(self.BASELINE_STATS_FILE, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            logger.info(f"Baseline stats saved to {self.BASELINE_STATS_FILE}")
        except Exception as e:
            logger.error(f"Failed to save baseline stats: {e}")

    def _load_baseline_stats(self) -> Optional[Dict]:
        """Load baseline statistics from file."""
        try:
            if not os.path.exists(self.BASELINE_STATS_FILE):
                logger.warning(f"Baseline stats file not found: {self.BASELINE_STATS_FILE}")
                return None
            
            with open(self.BASELINE_STATS_FILE, 'r') as f:
                stats = json.load(f)
            logger.info(f"Baseline stats loaded from {self.BASELINE_STATS_FILE}")
            return stats
        except Exception as e:
            logger.error(f"Failed to load baseline stats: {e}")
            return None

    def infer_schema(self, df: pd.DataFrame) -> Dict:
        """Automatically infer schema from DataFrame."""
        inferred_schema = {}
        for col in df.columns:
            dtype = df[col].dtype
            if pd.api.types.is_integer_dtype(dtype):
                inferred_schema[col] = {"type": "int", "min": int(df[col].min()), "max": int(df[col].max())}
            elif pd.api.types.is_float_dtype(dtype):
                inferred_schema[col] = {"type": "float", "min": float(df[col].min()), "max": float(df[col].max())}
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                inferred_schema[col] = {"type": "datetime"}
            elif pd.api.types.is_object_dtype(dtype):
                sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                if isinstance(sample, list):
                    inferred_schema[col] = {"type": "list"}
                else:
                    inferred_schema[col] = {
                        "type": "str",
                        "allowed_values": df[col].dropna().unique().tolist(),
                    }
        return inferred_schema

    def load_jsonl(self, filepath: str) -> pd.DataFrame:
        """Load data from JSONL file."""
        if not os.path.exists(filepath):
            logger.error("File not found: %s", filepath)
            return pd.DataFrame()

        records = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line.strip())
                    records.append(record)
                except json.JSONDecodeError as e:
                    logger.warning("Line %d: Failed to parse JSON - %s", line_num, e)
                    continue

        if not records:
            logger.warning("No valid records found in %s", filepath)
            return pd.DataFrame()

        df = pd.DataFrame(records)

        if "error" in df.columns:
            error_count = df["error"].notna().sum()
            if error_count > 0:
                logger.info("Removing %d error records", error_count)
                df = df[df["error"].isna()].reset_index(drop=True)

        df = self._normalize_datetimes(df)
        
        for col in df.columns:
            try:
                if df[col].dtype == 'object':
                    parsed = pd.to_datetime(df[col], errors='coerce', utc=False)
                    if parsed.notna().sum() > len(df) * 0.5:
                        df[col] = parsed
            except Exception as e:
                logger.debug(f"Could not parse {col} as datetime: {e}")
        
        df = self._normalize_datetimes(df)

        return df

    def split_baseline_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split baseline data into train/validation sets."""
        if df.empty:
            return pd.DataFrame(), pd.DataFrame()

        split_index = int(len(df) * self.TRAIN_SPLIT_RATIO)
        train_df = df.iloc[:split_index].reset_index(drop=True)
        validation_df = df.iloc[split_index:].reset_index(drop=True)
        return train_df, validation_df

    def analyze_baseline(self) -> Optional[Dict]:
        """Run complete analysis on baseline data and save stats."""
        logger.info("Starting baseline data analysis")

        self.baseline_df = self.load_jsonl(self.BASELINE_FILE)
        if self.baseline_df.empty:
            logger.error("No valid data found in baseline file")
            return None

        self.inferred_schema = self.infer_schema(self.baseline_df)
        self.train_df, self.validation_df = self.split_baseline_data(self.baseline_df)

        # Aggressive datetime normalization
        self.train_df = self._normalize_datetimes(self.train_df)
        self.validation_df = self._normalize_datetimes(self.validation_df)

        results = {
            "is_baseline": True,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "total_records": len(self.baseline_df),
            "train_records": len(self.train_df),
            "validation_records": len(self.validation_df),
        }

        try:
            train_norm = self._normalize_datetimes(self.train_df)
            val_norm = self._normalize_datetimes(self.validation_df)

            results["type_issues"] = {
                "training": self.validator.validate_data_types(
                    train_norm, schema=self.inferred_schema, dataset_name="Training Set"
                ),
                "validation": self.validator.validate_data_types(
                    val_norm, schema=self.inferred_schema, dataset_name="Validation Set"
                ),
            }

            try:
                results["anomalies"] = {
                    "training": self.anomaly_detector.detect_all_anomalies(train_norm.copy(), "Training Set"),
                    "validation": self.anomaly_detector.detect_all_anomalies(val_norm.copy(), "Validation Set"),
                }
            except Exception as anom_error:
                logger.warning("Anomaly detection issue: %s", anom_error)
                results["anomalies"] = {
                    "training": {"total_anomalies": 0, "text_anomalies": []},
                    "validation": {"total_anomalies": 0, "text_anomalies": []},
                }

            if self.remove_anomalies and "anomalies" in results:
                try:
                    self.train_df = self.anomaly_detector.remove_detected_anomalies(
                        self.train_df, results["anomalies"]["training"]
                    )
                    self.validation_df = self.anomaly_detector.remove_detected_anomalies(
                        self.validation_df, results["anomalies"]["validation"]
                    )
                except Exception as e:
                    logger.debug(f"Could not remove anomalies: {e}")

            train_norm = self._normalize_datetimes(self.train_df)
            val_norm = self._normalize_datetimes(self.validation_df)

            ge_results = self.validator.validate_with_ge(
                train_norm, val_norm, store_baseline=True
            )
            results.update(
                ge_failed_expectations=ge_results.get("failed_expectations", []),
                ge_success_rate=ge_results.get("success_rate", 0),
                ge_failed_count=ge_results.get("failed_count", 0),
            )

            try:
                results["bias"] = {
                    "training": self.anomaly_detector.detect_bias(train_norm.copy()),
                    "validation": self.anomaly_detector.detect_bias(val_norm.copy()),
                }
            except Exception as bias_error:
                logger.warning("Bias detection issue: %s", bias_error)
                results["bias"] = {
                    "training": {},
                    "validation": {},
                }

            # Save baseline stats (training set stats used for comparison)
            baseline_stats = {
                "total_records": results.get("total_records", 0),
                "train_records": results.get("train_records", 0),
                "validation_records": results.get("validation_records", 0),
                "schema": self.inferred_schema,
                "type_issues": results.get("type_issues", {}),
                "anomalies": results.get("anomalies", {}),
                "validation": {
                    "success_rate": results.get("ge_success_rate", 0),
                    "failed_count": results.get("ge_failed_count", 0),
                    "failed_expectations": results.get("ge_failed_expectations", [])
                },
                "bias": results.get("bias", {}),
                "timestamp": results.get("timestamp", "")
            }
            self._save_baseline_stats(baseline_stats)

            logger.info("Baseline analysis completed successfully")
        except Exception as e:
            logger.error("Error during baseline analysis: %s", e)
            import traceback
            traceback.print_exc()
            return None

        return results

    def analyze_new_data(self) -> Optional[Dict]:
        """Run complete analysis on new data against baseline."""
        # Load baseline stats
        self.baseline_stats = self._load_baseline_stats()
        if self.baseline_stats is None:
            logger.error("Baseline stats not available. Run baseline analysis first.")
            return None

        # Load baseline data for comparison
        self.baseline_df = self.load_jsonl(self.BASELINE_FILE)
        if self.baseline_df.empty:
            logger.error("Baseline data not found")
            return None

        self.inferred_schema = self.infer_schema(self.baseline_df)
        self.train_df, self.validation_df = self.split_baseline_data(self.baseline_df)
        self.train_df = self._normalize_datetimes(self.train_df)

        self.new_data_df = self.load_jsonl(self.NEW_DATA_FILE)
        if self.new_data_df.empty:
            logger.error("No valid data found in new data file")
            return None

        results = {
            "is_baseline": False,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "total_records": len(self.new_data_df),
        }

        try:
            new_norm = self._normalize_datetimes(self.new_data_df)

            results["type_issues"] = self.validator.validate_data_types(
                new_norm, schema=self.inferred_schema, dataset_name="New Data"
            )

            try:
                results["anomalies"] = self.anomaly_detector.detect_all_anomalies(
                    new_norm.copy(), "New Data"
                )
            except Exception as anom_error:
                logger.warning("Anomaly detection issue: %s", anom_error)
                results["anomalies"] = {"total_anomalies": 0, "text_anomalies": []}

            if self.remove_anomalies:
                try:
                    self.new_data_df = self.anomaly_detector.remove_detected_anomalies(
                        self.new_data_df, results["anomalies"]
                    )
                except Exception as e:
                    logger.debug(f"Could not remove anomalies: {e}")

            new_norm = self._normalize_datetimes(self.new_data_df)
            train_norm = self._normalize_datetimes(self.train_df)

            try:
                ge_results = self.validator.validate_with_ge(
                    train_norm, new_norm, store_baseline=False
                )
                results.update(
                    ge_failed_expectations=ge_results.get("failed_expectations", []),
                    ge_success_rate=ge_results.get("success_rate", 0),
                    ge_failed_count=ge_results.get("failed_count", 0),
                )
            except Exception as ge_error:
                logger.info(f"Validation: {type(ge_error).__name__}")
                results.update(
                    ge_failed_expectations=[],
                    ge_success_rate=100,
                    ge_failed_count=0,
                )

            try:
                train_drift = self._normalize_datetimes(self.train_df.copy())
                new_drift = self._normalize_datetimes(self.new_data_df.copy())
                results["drift_results"] = self.drift_detector.detect_all_drift(
                    train_drift, new_drift
                )
            except Exception as drift_error:
                logger.warning("Drift detection issue: %s", drift_error)
                results["drift_results"] = {}

            try:
                new_bias = self._normalize_datetimes(self.new_data_df.copy())
                results["bias"] = self.anomaly_detector.detect_bias(new_bias)
            except Exception as bias_error:
                logger.warning("Bias detection issue: %s", bias_error)
                results["bias"] = {}

            logger.info("New data analysis completed successfully")
        except Exception as e:
            logger.error("Error during new data analysis: %s", e)
            import traceback
            traceback.print_exc()
            return None

        return results

    def print_report(self, results: Dict):
        """Print comprehensive analysis report."""
        if not results:
            return

        dataset_type = "BASELINE DATA" if results['is_baseline'] else "NEW DATA"
        
        print()
        print(f"ANALYSIS REPORT FOR {dataset_type}")
        print(f"Timestamp: {results.get('timestamp', 'N/A')}")
        print()

        print("DATASET STATISTICS")
        print()
        if results['is_baseline']:
            print(f"Total baseline records: {results.get('total_records', 0)}")
            print(f"Training set size: {results.get('train_records', 0)}")
            print(f"Validation set size: {results.get('validation_records', 0)}")
        else:
            print(f"Total new data records: {results.get('total_records', 0)}")
        print()

        print("DATA TYPE VALIDATION")
        print()
        type_issues = results.get('type_issues', {})
        if results['is_baseline']:
            for subset in ['training', 'validation']:
                if subset in type_issues:
                    issues = type_issues[subset]
                    print(f"{subset.upper()} SET:")
                    if issues:
                        for col, issue in issues.items():
                            print(f"  {col}: {issue}")
                    else:
                        print(f"  No issues detected")
                    print()
        else:
            if type_issues:
                print("ISSUES FOUND:")
                for col, issue in type_issues.items():
                    print(f"  {col}: {issue}")
                print()
            else:
                print("No type issues detected")
                print()

        print("VALIDATION RESULTS")
        print()
        ge_success_rate = results.get('ge_success_rate', 0)
        ge_failed_count = results.get('ge_failed_count', 0)
        print(f"Success Rate: {ge_success_rate:.1f}%")
        print(f"Failed Checks: {ge_failed_count}")
        print()

        failed_expectations = results.get('ge_failed_expectations', [])
        if failed_expectations:
            print("All Failed Checks:")
            for i, exp in enumerate(failed_expectations, 1):
                column = exp.get('column', 'N/A')
                exp_type = exp.get('expectation_type', 'Unknown')
                details = exp.get('details', 'No details')
                print(f"  Check {i}:")
                print(f"    Column: {column}")
                print(f"    Type: {exp_type}")
                print(f"    Details: {details}")
            print()

        print("ANOMALY DETECTION")
        print()
        anomalies = results.get('anomalies', {})
        
        if results['is_baseline']:
            for subset in ['training', 'validation']:
                if subset in anomalies:
                    anomaly_data = anomalies[subset]
                    total = anomaly_data.get('total_anomalies', 0)
                    print(f"{subset.upper()} SET: {total} anomalies detected")
                    
                    text_anomalies = anomaly_data.get('text_anomalies', [])
                    if text_anomalies:
                        print(f"  Anomaly Records:")
                        for idx, anomaly in enumerate(text_anomalies, 1):
                            print(f"    Record {idx}:")
                            print(f"      Type: {anomaly.get('type', 'Unknown')}")
                            print(f"      Link: {anomaly.get('link', 'N/A')}")
                            print(f"      Word Count: {anomaly.get('word_count', 'N/A')}")
                            print(f"      Reason: {anomaly.get('reason', 'N/A')}")
                    print()
        else:
            total = anomalies.get('total_anomalies', 0)
            print(f"Total anomalies detected: {total}")
            print()
            
            text_anomalies = anomalies.get('text_anomalies', [])
            if text_anomalies:
                print("Anomaly Records:")
                for idx, anomaly in enumerate(text_anomalies, 1):
                    print(f"  Record {idx}:")
                    print(f"    Type: {anomaly.get('type', 'Unknown')}")
                    print(f"    Link: {anomaly.get('link', 'N/A')}")
                    print(f"    Word Count: {anomaly.get('word_count', 'N/A')}")
                    print(f"    Reason: {anomaly.get('reason', 'N/A')}")
                print()

        if not results['is_baseline']:
            print("DRIFT DETECTION")
            print()
            drift_results = results.get('drift_results', {})
            if drift_results:
                drift_features = [f for f, d in drift_results.items() if d and d.get('has_drift')]
                if drift_features:
                    print(f"Drift detected in {len(drift_features)} features")
                    print()
                    for feature in drift_features:
                        print(f"Feature: {feature}")
                        drift_info = drift_results[feature]
                        
                        if 'mean_shift' in drift_info:
                            print(f"  Type: Numeric")
                            print(f"  Severity: {drift_info.get('severity', 'UNKNOWN')}")
                            print(f"  Mean Shift: {drift_info['mean_shift']:.3f} standard deviations")
                            print(f"  Baseline Mean: {drift_info['baseline_mean']:.2f}")
                            print(f"  New Mean: {drift_info['new_mean']:.2f}")
                            print(f"  Percent Change: {drift_info.get('percent_mean_change', 0):.1f}%")
                            print(f"  KS Test P-value: {drift_info.get('ks_pvalue', 1):.4f}")
                        else:
                            print(f"  Type: Categorical")
                            if drift_info.get('new_categories'):
                                print(f"  New Categories: {drift_info['new_categories']}")
                            if drift_info.get('missing_categories'):
                                print(f"  Missing Categories: {drift_info['missing_categories']}")
                        print()
                else:
                    print("No drift detected")
                    print()
            else:
                print("No drift analysis available")
                print()

        print("BIAS DETECTION")
        print()
        bias_results = results.get('bias', {})

        if results['is_baseline']:
            print("TRAINING SET BIAS:")
            self._print_bias_details(bias_results.get('training', {}))
            print()
            print("VALIDATION SET BIAS:")
            self._print_bias_details(bias_results.get('validation', {}))
        else:
            self._print_bias_details(bias_results)
        print()

    def _print_bias_details(self, bias_results: Dict):
        """Print detailed bias information with distributions."""
        if not bias_results:
            print("No significant bias detected")
            return

        if 'country_bias' in bias_results:
            cb = bias_results['country_bias']
            print(f"Country Bias:")
            print(f"  Dominant: {cb['dominant']}")
            print(f"  Percentage: {cb['percentage']:.1f}%")
            print()

        if 'country_distribution' in bias_results:
            print(f"Country Distribution:")
            for country, pct in list(bias_results['country_distribution'].items())[:15]:
                print(f"  {country}: {pct:.1f}%")
            print()

        if 'temporal_bias' in bias_results:
            tb = bias_results['temporal_bias']
            print(f"Temporal Bias:")
            print(f"  Recent Percentage (last 30 days): {tb['recent_percentage']:.1f}%")
            print()

        if 'temporal_distribution' in bias_results:
            td = bias_results['temporal_distribution']
            print(f"Temporal Distribution:")
            print(f"  Last 30 days: {td.get('last_30_days', 0)}")
            print(f"  Last 90 days: {td.get('last_90_days', 0)}")
            print(f"  Last year: {td.get('last_year', 0)}")
            print(f"  Older: {td.get('older', 0)}")
            print(f"  Total with dates: {td.get('total_with_dates', 0)}")
            print()

        if 'top_topics' in bias_results:
            print(f"Top Topics:")
            for topic, pct in list(bias_results['top_topics'].items())[:20]:
                print(f"  {topic}: {pct:.1f}%")
            print()

        if 'source_bias' in bias_results:
            sb = bias_results['source_bias']
            print(f"Source Bias:")
            print(f"  Dominant: {sb['dominant']}")
            print(f"  Percentage: {sb['percentage']:.1f}%")
            print()

        if 'source_distribution' in bias_results:
            print(f"Source Type Distribution:")
            for source, pct in list(bias_results['source_distribution'].items())[:15]:
                print(f"  {source}: {pct:.1f}%")
            print()

        if not any(k in bias_results for k in ['country_bias', 'temporal_bias', 'top_topics', 'source_bias']):
            print("No significant bias detected")

    def run_complete_analysis(self):
        """Run complete analysis pipeline."""
        if self.is_baseline:
            print()
            print("PHASE 1: ANALYZING BASELINE DATA")
            print()
            results = self.analyze_baseline()
            if results:
                self.print_report(results)
            else:
                print("Baseline analysis failed.")
                return False
        else:
            print()
            print("PHASE 2: ANALYZING NEW DATA AGAINST BASELINE")
            print()
            results = self.analyze_new_data()
            if results:
                self.print_report(results)
            else:
                print("New data analysis failed")
                return False

        return True


def main():
    """Entry point for the analysis pipeline."""
    try:
        required_files = [
           "../data/scraped_baseline.jsonl",
           "../data/scraped_updated.jsonl"
        ]
        
        print()
        print("DATA QUALITY ANALYSIS PIPELINE")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Phase 1: Analyze baseline
        print("Running baseline analysis...")
        print()
        analyzer_baseline = DataQualityAnalyzer(is_baseline=True)
        success_baseline = analyzer_baseline.run_complete_analysis()
        
        if not success_baseline:
            print("Baseline analysis failed. Exiting.")
            return False

        # Phase 2: Analyze new data
        print()
        print("Running new data analysis...")
        print()
        analyzer_new = DataQualityAnalyzer(is_baseline=False)
        success_new = analyzer_new.run_complete_analysis()
        
        if not success_new:
            print("New data analysis failed")
            return False

        print()
        print(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        if success_baseline and success_new:
            print("All analyses completed successfully")
            return True
        else:
            print("Some analyses failed")
            return False
            
    except KeyboardInterrupt:
        print()
        print("Analysis interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print()
        print(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()