"""Drift detection module for monitoring data distribution changes.

Detects drift in numerical, categorical, and topics features using statistical tests.
"""

import logging
from typing import Dict, Optional

import pandas as pd
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class DriftDetector:
    """Detects drift in data distributions."""

    DRIFT_THRESHOLD = 0.1

    def _normalize_datetimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize datetime columns by converting to string and back.
        
        This eliminates any offset-naive and offset-aware datetime mismatches.
        """
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

    def calculate_numerical_drift(
        self, baseline_data: pd.Series, new_data: pd.Series
    ) -> Optional[Dict]:
        """Calculate drift for numerical features using KS test."""
        if baseline_data is None or new_data is None:
            return None

        try:
            baseline_clean = pd.to_numeric(baseline_data, errors='coerce').dropna()
            new_clean = pd.to_numeric(new_data, errors='coerce').dropna()

            if len(baseline_clean) == 0 or len(new_clean) == 0:
                return None

            ks_stat, ks_p = stats.ks_2samp(baseline_clean, new_clean)

            baseline_mean = baseline_clean.mean()
            baseline_std = baseline_clean.std()
            new_mean = new_clean.mean()
            new_std = new_clean.std()

            mean_shift = abs(new_mean - baseline_mean) / (baseline_std + 1e-7)
            std_ratio = new_std / (baseline_std + 1e-7)

            drift_info = {
                'mean_shift': mean_shift,
                'std_ratio': std_ratio,
                'baseline_mean': baseline_mean,
                'new_mean': new_mean,
                'baseline_std': baseline_std,
                'new_std': new_std,
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_p,
                'percent_mean_change': ((new_mean - baseline_mean) / baseline_mean * 100)
                if baseline_mean != 0 else 0
            }

            drift_info['has_drift'] = (
                mean_shift > self.DRIFT_THRESHOLD or
                abs(std_ratio - 1.0) > self.DRIFT_THRESHOLD or
                ks_p < 0.05
            )
            return drift_info

        except Exception as e:
            logger.error(f"Numerical drift calculation failed: {e}")
            return None

    def _safe_chi2_test(
        self, baseline_counts: pd.Series, new_counts: pd.Series
    ) -> tuple:
        """Perform chi-square test safely with proper frequency normalization."""
        try:
            # Get all categories
            all_categories = baseline_counts.index.union(new_counts.index)
            
            # Align frequencies
            baseline_freq = baseline_counts.reindex(all_categories, fill_value=0)
            new_freq = new_counts.reindex(all_categories, fill_value=0)
            
            # Convert to proportions
            baseline_prop = baseline_freq / (baseline_freq.sum() + 1e-10)
            new_prop = new_freq / (new_freq.sum() + 1e-10)
            
            # Scale to same total for chi-square
            total_count = max(baseline_freq.sum(), new_freq.sum())
            baseline_scaled = baseline_prop * total_count
            new_scaled = new_prop * total_count
            
            # Ensure no zero/negative values
            baseline_scaled = np.maximum(baseline_scaled, 1e-10)
            new_scaled = np.maximum(new_scaled, 1e-10)
            
            chi2, p_value = stats.chisquare(f_obs=new_scaled, f_exp=baseline_scaled)
            return chi2, p_value
        except Exception as e:
            logger.debug(f"Chi-square test failed: {e}")
            return None, None

    def calculate_categorical_drift(
        self, baseline_data: pd.Series, new_data: pd.Series
    ) -> Optional[Dict]:
        """Calculate drift for categorical features."""
        if baseline_data is None or new_data is None:
            return None

        try:
            baseline_clean = baseline_data.dropna()
            new_clean = new_data.dropna()
            
            if len(baseline_clean) == 0 or len(new_clean) == 0:
                return {
                    'has_drift': False,
                    'new_categories': [],
                    'missing_categories': [],
                    'total_categories_baseline': 0,
                    'total_categories_new': 0,
                    'top_categories_baseline': {},
                    'top_categories_new': {}
                }

            baseline_counts = baseline_clean.value_counts()
            new_counts = new_clean.value_counts()

            drift_info = {
                'new_categories': list(set(new_counts.index) - set(baseline_counts.index)),
                'missing_categories': list(set(baseline_counts.index) - set(new_counts.index)),
                'total_categories_baseline': len(baseline_counts),
                'total_categories_new': len(new_counts),
                'top_categories_baseline': baseline_counts.head(5).to_dict(),
                'top_categories_new': new_counts.head(5).to_dict()
            }

            # Perform chi-square test if multiple categories exist
            if len(baseline_counts) > 1 and len(new_counts) > 1:
                chi2, p_value = self._safe_chi2_test(baseline_counts, new_counts)
                if chi2 is not None:
                    drift_info['chi2_statistic'] = chi2
                    drift_info['chi2_pvalue'] = p_value
                    drift_info['has_drift'] = p_value < 0.05
                else:
                    drift_info['has_drift'] = (
                        len(drift_info['new_categories']) > 0 or 
                        len(drift_info['missing_categories']) > 0
                    )
            else:
                drift_info['has_drift'] = (
                    len(drift_info['new_categories']) > 0 or 
                    len(drift_info['missing_categories']) > 0
                )

            return drift_info

        except Exception as e:
            logger.error(f"Categorical drift calculation failed: {e}")
            return {
                'has_drift': False,
                'new_categories': [],
                'missing_categories': [],
                'total_categories_baseline': 0,
                'total_categories_new': 0,
                'top_categories_baseline': {},
                'top_categories_new': {}
            }

    def calculate_topics_drift(
        self, baseline_df: pd.DataFrame, new_df: pd.DataFrame
    ) -> Dict:
        """Calculate drift for topics feature."""
        try:
            baseline_topics_col = baseline_df.get('topics', pd.Series(dtype=object))
            new_topics_col = new_df.get('topics', pd.Series(dtype=object))
            
            baseline_topics = []
            for lst in baseline_topics_col.dropna():
                if isinstance(lst, list):
                    baseline_topics.extend([str(t).lower() for t in lst])
            
            new_topics = []
            for lst in new_topics_col.dropna():
                if isinstance(lst, list):
                    new_topics.extend([str(t).lower() for t in lst])

            if not baseline_topics and not new_topics:
                return {
                    'new_topics': [],
                    'missing_topics': [],
                    'total_topics_baseline': 0,
                    'total_topics_new': 0,
                    'top_topics_baseline': {},
                    'top_topics_new': {},
                    'has_drift': False
                }

            baseline_dist = pd.Series(baseline_topics).value_counts()
            new_dist = pd.Series(new_topics).value_counts()

            new_unique = set(new_dist.index) - set(baseline_dist.index)
            missing = set(baseline_dist.index) - set(new_dist.index)
            common = set(baseline_dist.index) & set(new_dist.index)

            drift_info = {
                'new_topics': list(new_unique)[:10],
                'missing_topics': list(missing)[:10],
                'total_topics_baseline': len(baseline_dist),
                'total_topics_new': len(new_dist),
                'top_topics_baseline': baseline_dist.head(5).to_dict(),
                'top_topics_new': new_dist.head(5).to_dict(),
                'has_drift': False
            }

            # Perform chi-square test on common topics
            if len(common) > 1:
                chi2, p_value = self._safe_chi2_test(baseline_dist, new_dist)
                if chi2 is not None:
                    drift_info['chi2_statistic'] = chi2
                    drift_info['chi2_pvalue'] = p_value
                    drift_info['has_drift'] = p_value < 0.05
                else:
                    drift_info['has_drift'] = len(new_unique) > 0 or len(missing) > 0
            else:
                drift_info['has_drift'] = len(new_unique) > 0 or len(missing) > 0

            return drift_info

        except Exception as e:
            logger.error(f"Topics drift calculation failed: {e}")
            return {
                'new_topics': [],
                'missing_topics': [],
                'total_topics_baseline': 0,
                'total_topics_new': 0,
                'top_topics_baseline': {},
                'top_topics_new': {},
                'has_drift': False,
                'error': str(e)
            }
        
    def detect_all_drift(
        self, baseline_df: pd.DataFrame, new_df: pd.DataFrame
    ) -> Dict:
        """Detect drift for numerical, categorical, and topics features."""
        # Normalize datetimes first to prevent comparison errors
        try:
            baseline_df = baseline_df.copy()
            new_df = new_df.copy()
            baseline_df = self._normalize_datetimes(baseline_df)
            new_df = self._normalize_datetimes(new_df)
        except Exception as e:
            logger.warning(f"Could not normalize datetimes: {e}")
        
        drift_results = {}

        # Numerical features
        for feature in ['word_count', 'token_count']:
            if feature in baseline_df.columns and feature in new_df.columns:
                try:
                    drift_info = self.calculate_numerical_drift(
                        baseline_df[feature], new_df[feature]
                    )
                    if drift_info:
                        drift_results[feature] = drift_info
                except Exception as e:
                    logger.warning(f"Numerical drift for {feature} failed: {e}")

        # Categorical features
        for feature in ['country', 'source_type']:
            if feature in baseline_df.columns and feature in new_df.columns:
                try:
                    drift_info = self.calculate_categorical_drift(
                        baseline_df[feature], new_df[feature]
                    )
                    if drift_info:
                        drift_results[feature] = drift_info
                except Exception as e:
                    logger.warning(f"Categorical drift for {feature} failed: {e}")

        # Topics feature
        if 'topics' in baseline_df.columns and 'topics' in new_df.columns:
            try:
                drift_info = self.calculate_topics_drift(baseline_df, new_df)
                if drift_info:
                    drift_results['topics'] = drift_info
            except Exception as e:
                logger.warning(f"Topics drift detection failed: {e}")

        return drift_results