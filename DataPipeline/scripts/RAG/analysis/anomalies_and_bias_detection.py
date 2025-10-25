"""Anomaly and bias detection module"""

import logging
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Detect anomalies and biases using pandas only."""
    
    LOW_WORD_THRESHOLD = 100
    HIGH_WORD_THRESHOLD = 5000
    
    def detect_all_anomalies(
        self, 
        df: pd.DataFrame, 
        dataset_name: str
    ) -> Dict:
        """Detect anomalies in a DataFrame using simple rules.
        
        Args:
            df: DataFrame to analyze
            dataset_name: Name of the dataset for logging
            
        Returns:
            Dictionary containing anomaly information
        """
        anomalies = []
        
        # Word count anomalies
        if 'word_count' in df.columns:
            for idx in df.index:
                value = df.at[idx, 'word_count']
                
                # Check for anomalies
                try:
                    if pd.isna(value) or np.isinf(value):
                        anomalies.append({
                            'expectation': 'word_count_out_of_bounds',
                            'column': 'word_count',
                            'index': idx,
                            'value': value
                        })
                    elif value < self.LOW_WORD_THRESHOLD or value > self.HIGH_WORD_THRESHOLD:
                        anomalies.append({
                            'expectation': 'word_count_out_of_bounds',
                            'column': 'word_count',
                            'index': idx,
                            'value': value
                        })
                except (TypeError, ValueError):
                    # If comparison fails, it's an anomaly
                    anomalies.append({
                        'expectation': 'word_count_out_of_bounds',
                        'column': 'word_count',
                        'index': idx,
                        'value': value
                    })
        
        # Text anomalies
        if 'text' in df.columns:
            for idx in df.index:
                value = df.at[idx, 'text']
                
                # Check for null
                if value is None or (isinstance(value, float) and pd.isna(value)):
                    anomalies.append({
                        'expectation': 'text_null_or_blank',
                        'column': 'text',
                        'index': idx,
                        'value': value
                    })
                # Check for blank string
                elif isinstance(value, str) and value.strip() == "":
                    anomalies.append({
                        'expectation': 'text_null_or_blank',
                        'column': 'text',
                        'index': idx,
                        'value': value
                    })
        
        # Topics anomalies
        if 'topics' in df.columns:
            for idx in df.index:
                value = df.at[idx, 'topics']
                is_valid = self._is_valid_topics_value(value)
                
                if not is_valid:
                    anomalies.append({
                        'expectation': 'topics_not_list_or_null',
                        'column': 'topics',
                        'index': idx,
                        'value': value
                    })
        
        return {
            'dataset': dataset_name,
            'text_anomalies': anomalies,
            'total_anomalies': len(anomalies),
        }
    
    def _is_valid_topics_value(self, value: Any) -> bool:
        """Check if a topics value is valid (list or array).
        
        Args:
            value: The value to check
            
        Returns:
            True if value is a valid list/array, False otherwise
        """
        # Lists and arrays are valid (even if empty)
        if isinstance(value, (list, np.ndarray)):
            return True
        
        # None is invalid
        if value is None:
            return False
        
        # NaN is invalid
        try:
            if isinstance(value, float) and pd.isna(value):
                return False
        except:
            pass
        
        # Everything else is invalid
        return False
    
    def remove_detected_anomalies(
        self, 
        df: pd.DataFrame, 
        anomalies: Dict
    ) -> pd.DataFrame:
        """Remove rows flagged as anomalies.
        
        Args:
            df: Original DataFrame
            anomalies: Dictionary containing anomaly information
            
        Returns:
            DataFrame with anomalous rows removed
        """
        if not anomalies or 'rows' not in anomalies:
            return df
        
        return df.drop(index=anomalies['rows']).reset_index(drop=True)
    
    def detect_bias(self, df: pd.DataFrame) -> Dict:
        """Simple bias detection using pandas.
        
        Args:
            df: DataFrame to analyze for bias
            
        Returns:
            Dictionary containing bias metrics
        """
        bias = {}
        
        # Country bias detection
        if 'country' in df.columns:
            counts = df['country'].value_counts(normalize=True) * 100
            if not counts.empty:
                bias['country_bias'] = {
                    'dominant': counts.idxmax(),
                    'percentage': counts.iloc[0]
                }
        
        # Source type bias detection
        if 'source_type' in df.columns:
            counts = df['source_type'].value_counts(normalize=True) * 100
            if not counts.empty:
                bias['source_bias'] = {
                    'dominant': counts.idxmax(),
                    'percentage': counts.iloc[0]
                }
        
        # Topics analysis
        if 'topics' in df.columns:
            all_topics = []
            
            for idx in df.index:
                value = df.at[idx, 'topics']
                
                # Skip None values
                if value is None:
                    continue
                
                # Handle lists
                if isinstance(value, list):
                    all_topics.extend(value)
                # Handle numpy arrays
                elif isinstance(value, np.ndarray):
                    all_topics.extend(value.tolist())
                # Skip NaN values
                elif isinstance(value, float):
                    try:
                        if pd.isna(value):
                            continue
                    except:
                        continue
                    # If it's a non-NaN float, that's weird but add it
                    all_topics.append(value)
                else:
                    # For any other single value
                    all_topics.append(value)
            
            if all_topics:
                topic_series = pd.Series(all_topics)
                bias['top_topics'] = (
                    topic_series.value_counts()
                    .head(5)
                    .to_dict()
                )
        
        # Temporal distribution
        if 'publish_date' in df.columns:
            valid_dates = pd.to_datetime(
                df['publish_date'], 
                errors='coerce'
            ).dropna()
            
            if not valid_dates.empty:
                now = pd.Timestamp.now()
                days_diff = (now - valid_dates).dt.days
                
                bias['temporal_distribution'] = {
                    'total_with_dates': len(valid_dates),
                    'last_30_days': int((days_diff <= 30).sum()),
                    'last_90_days': int((days_diff <= 90).sum()),
                    'last_year': int((days_diff <= 365).sum()),
                    'older': int((days_diff > 365).sum())
                }
                
                # Check for temporal bias
                recent_pct = (days_diff <= 30).sum() / len(valid_dates) * 100
                if recent_pct > 80:
                    bias['temporal_bias'] = {
                        'recent_percentage': float(recent_pct)
                    }
        
        return bias