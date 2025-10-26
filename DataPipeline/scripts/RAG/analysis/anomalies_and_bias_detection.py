"""Simple TFDV-style anomaly detection using baseline statistics."""

import logging
from typing import Dict, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Detect anomalies by comparing against baseline statistics."""

    def __init__(self):
        """Initialize with hardcoded word_count thresholds."""
        self.min_word_count = 100
        self.max_word_count = 10000

    def compute_baseline_stats(self, df: pd.DataFrame) -> Dict:
        """Compute baseline statistics for each column.

        Args:
            df: Training DataFrame

        Returns:
            Dict with stats for each column (min, max, lengths, null %)
        """
        stats = {}

        for col in df.columns:
            if col == 'error':
                continue

            col_stats = {}

            # Numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    col_stats = {
                        "type": "numeric",
                        "min": float(non_null.min()),
                        "max": float(non_null.max()),
                        "mean": float(non_null.mean()),
                        "std": float(non_null.std(ddof=0)),
                        "null_pct": float((df[col].isna().sum() / len(df)) * 100)
                    }

            # Datetime columns
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    col_stats = {
                        "type": "datetime",
                        "min": str(non_null.min()),
                        "max": str(non_null.max()),
                        "null_pct": float(
                            (df[col].isna().sum() / len(df)) * 100
                        )
                    }

            # Object columns
            elif pd.api.types.is_object_dtype(df[col]):
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    sample = non_null.iloc[0]

                    # List columns
                    if isinstance(sample, list):
                        list_lengths = [
                            len(v) for v in non_null if isinstance(v, list)
                        ]
                        col_stats = {
                            "type": "list",
                            "min_len": int(min(list_lengths))
                            if list_lengths else 0,
                            "max_len": int(max(list_lengths))
                            if list_lengths else 0,
                            "avg_len": float(np.mean(list_lengths))
                            if list_lengths else 0,
                            "null_pct": float(
                                (df[col].isna().sum() / len(df)) * 100
                            )
                        }

                    # Text columns
                    else:
                        str_lengths = [len(str(v)) for v in non_null]
                        col_stats = {
                            "type": "text",
                            "min_len": int(min(str_lengths))
                            if str_lengths else 0,
                            "max_len": int(max(str_lengths))
                            if str_lengths else 0,
                            "avg_len": float(np.mean(str_lengths))
                            if str_lengths else 0,
                            "null_pct": float(
                                (df[col].isna().sum() / len(df)) * 100
                            )
                        }

            if col_stats:
                stats[col] = col_stats

        return stats

    def detect_anomalous_records(
        self,
        df: pd.DataFrame,
        baseline_stats: Dict,
        dataset_name: str
    ) -> List[Dict]:
        """Find records that violate baseline statistics.

        Args:
            df: DataFrame to check (validation or new data)
            baseline_stats: Stats from training data
            dataset_name: Name for logging

        Returns:
            List of anomalous records with violations
        """
        anomalous_records = []

        for idx in df.index:
            violations = []
            link = df.at[idx, 'link'] if 'link' in df.columns else 'N/A'

            # Check each column against baseline
            for col, stats in baseline_stats.items():
                if col not in df.columns:
                    continue

                value = df.at[idx, col]
                col_type = stats.get('type')

                # Numeric columns
                if col_type == 'numeric':
                    if pd.notna(value):
                        if value < stats['min']:
                            violations.append(
                                f"{col}={value:.2f} < min={stats['min']:.2f}"
                            )
                        elif value > stats['max']:
                            violations.append(
                                f"{col}={value:.2f} > max={stats['max']:.2f}"
                            )
                    else:
                        if stats['null_pct'] < 5.0:
                            violations.append(
                                f"{col} is null "
                                f"(baseline {stats['null_pct']:.1f}% nulls)"
                            )

                # Text columns
                elif col_type == 'text':
                    is_null = (
                        value is None or
                        (isinstance(value, float) and pd.isna(value))
                    )

                    if not is_null:
                        length = len(str(value))
                        if length < stats['min_len'] * 0.5:
                            violations.append(
                                f"{col} length={length} < "
                                f"min={stats['min_len']}"
                            )
                        elif length > stats['max_len'] * 2:
                            violations.append(
                                f"{col} length={length} > "
                                f"max={stats['max_len']}"
                            )
                    else:
                        if stats['null_pct'] < 5.0:
                            violations.append(
                                f"{col} is null "
                                f"(baseline {stats['null_pct']:.1f}% nulls)"
                            )

                # List columns
                elif col_type == 'list':
                    if isinstance(value, list):
                        length = len(value)
                        if (stats['max_len'] > 0 and
                                length > stats['max_len'] * 2):
                            violations.append(
                                f"{col} has {length} items > "
                                f"max={stats['max_len']}"
                            )
                    else:
                        is_null = (
                            value is None or
                            (isinstance(value, float) and pd.isna(value))
                        )
                        if is_null and stats['null_pct'] < 5.0:
                            violations.append(
                                f"{col} is null "
                                f"(baseline {stats['null_pct']:.1f}% nulls)"
                            )

            # Check hardcoded word_count thresholds
            if 'word_count' in df.columns:
                word_count = df.at[idx, 'word_count']
                if pd.notna(word_count):
                    if word_count < self.min_word_count:
                        violations.append(
                            f"word_count={word_count} < "
                            f"min_threshold={self.min_word_count}"
                        )
                    elif word_count > self.max_word_count:
                        violations.append(
                            f"word_count={word_count} > "
                            f"max_threshold={self.max_word_count}"
                        )

            # Add record if it has violations
            if violations:
                anomalous_records.append({
                    'index': int(idx),
                    'link': link,
                    'violations': violations
                })

        logger.info(
            f"{dataset_name}: Found {len(anomalous_records)} "
            f"anomalous records"
        )
        return anomalous_records

    def detect_completeness_issues(
        self,
        df: pd.DataFrame,
        dataset_name: str
    ) -> Dict:
        """Detect missing/empty critical fields (topics, text, title).

        Args:
            df: DataFrame to check
            dataset_name: Name for logging

        Returns:
            Dict with lists of records for each issue type
        """
        issues = {}

        # Only check: topics, text, title
        checks = {
            'topics': 'MISSING_TOPICS',
            'text': 'EMPTY_TEXT',
            'title': 'MISSING_TITLE'
        }

        for col, issue_type in checks.items():
            if col not in df.columns:
                continue

            missing = []
            for idx in df.index:
                value = df.at[idx, col]
                link = (
                    df.at[idx, 'link'] if 'link' in df.columns else 'N/A'
                )

                is_null = (
                    value is None or
                    (isinstance(value, float) and pd.isna(value))
                )

                if is_null:
                    missing.append({
                        'index': int(idx),
                        'link': link,
                        'reason': f'{col} is null'
                    })
                elif isinstance(value, str) and len(value.strip()) == 0:
                    missing.append({
                        'index': int(idx),
                        'link': link,
                        'reason': f'{col} is empty'
                    })
                elif isinstance(value, list) and len(value) == 0:
                    missing.append({
                        'index': int(idx),
                        'link': link,
                        'reason': f'{col} is empty list'
                    })

            if missing:
                issues[issue_type] = missing

        return issues

    def detect_bias(self, df: pd.DataFrame) -> Dict:
        """Detect bias in the data.

        Args:
            df: DataFrame to analyze

        Returns:
            Dict containing bias metrics
        """
        bias = {}

        # Country bias
        if 'country' in df.columns:
            counts = df['country'].value_counts(normalize=True) * 100
            if not counts.empty:
                bias['country_bias'] = {
                    'dominant': counts.idxmax(),
                    'percentage': float(counts.iloc[0])
                }

        # Source type bias
        if 'source_type' in df.columns:
            counts = df['source_type'].value_counts(normalize=True) * 100
            if not counts.empty:
                bias['source_bias'] = {
                    'dominant': counts.idxmax(),
                    'percentage': float(counts.iloc[0])
                }

        # Topics
        if 'topics' in df.columns:
            all_topics = []
            for idx in df.index:
                value = df.at[idx, 'topics']
                if isinstance(value, list):
                    all_topics.extend(value)

            if all_topics:
                topic_series = pd.Series(all_topics)
                bias['top_topics'] = (
                    topic_series.value_counts().head(5).to_dict()
                )

        # Temporal distribution
        if 'publish_date' in df.columns:
            try:
                # Parse and clean dates
                date_series = pd.to_datetime(
                    df['publish_date'],
                    errors='coerce'
                )
                valid_dates = date_series.dropna()

                if len(valid_dates) > 0:
                    # Remove timezone if present
                    if valid_dates.dt.tz is not None:
                        valid_dates = valid_dates.dt.tz_localize(None)
                    
                    # Get current time as timezone-naive
                    now = pd.Timestamp.now()
                    
                    # Calculate days difference
                    days_diff = (now - valid_dates).dt.days

                    bias['temporal_distribution'] = {
                        'total_with_dates': int(len(valid_dates)),
                        'last_30_days': int((days_diff <= 30).sum()),
                        'last_90_days': int((days_diff <= 90).sum()),
                        'last_year': int((days_diff <= 365).sum()),
                        'older': int((days_diff > 365).sum())
                    }
            except Exception as e:
                logger.debug(
                    f"Skipping temporal distribution: {e}"
                )

        return bias