"""Comprehensive tests for DriftDetector module."""

import pandas as pd
import pytest
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../scripts/RAG/analysis'))

from drift import DriftDetector


@pytest.fixture
def sample_data():
    """Create baseline sample data."""
    return pd.DataFrame({
        'word_count': [100, 200, 150, 120, 180, 160, 140, 190, 110, 170],
        'token_count': [120, 250, 180, 150, 220, 190, 170, 230, 130, 200],
        'country': ['USA', 'USA', 'Canada', 'USA', 'UK', 'Canada', 'USA', 'Canada', 'USA', 'UK'],
        'source_type': ['blog', 'news', 'blog', 'news', 'blog', 'news', 'blog', 'news', 'blog', 'news'],
        'topics': [
            ['technology', 'ai'],
            ['politics', 'economy'],
            ['health', 'science'],
            ['technology', 'science'],
            ['politics', 'ai'],
            ['health', 'economy'],
            ['technology', 'economy'],
            ['science', 'ai'],
            ['health', 'politics'],
            ['economy', 'technology']
        ],
        'publish_date': pd.date_range(start='2025-01-01', periods=10)
    })


@pytest.fixture
def new_data():
    """Create new data with some drift."""
    return pd.DataFrame({
        'word_count': [110, 210, 140, 130, 190, 170, 150, 200, 120, 180],
        'token_count': [130, 240, 175, 160, 210, 200, 180, 220, 140, 210],
        'country': ['USA', 'Canada', 'USA', 'Mexico', 'UK', 'Canada', 'USA', 'Canada', 'USA', 'Mexico'],
        'source_type': ['blog', 'blog', 'news', 'social', 'blog', 'news', 'social', 'news', 'blog', 'social'],
        'topics': [
            ['technology', 'ml'],
            ['politics', 'trade'],
            ['health', 'vaccine'],
            ['technology', 'science'],
            ['ai', 'data'],
            ['health', 'economy'],
            ['technology', 'blockchain'],
            ['science', 'research'],
            ['health', 'politics'],
            ['trade', 'tech']
        ],
        'publish_date': pd.date_range(start='2025-02-01', periods=10)
    })


@pytest.fixture
def high_drift_data():
    """Create data with high drift."""
    return pd.DataFrame({
        'word_count': [5000, 4500, 5500, 4800, 5200, 5100, 4900, 5300, 5000, 4700],
        'token_count': [6000, 5500, 6500, 5800, 6200, 6100, 5900, 6300, 6000, 5700],
        'country': ['Mexico', 'Brazil', 'Argentina', 'Mexico', 'Brazil', 'Argentina', 'Mexico', 'Brazil', 'Argentina', 'Chile'],
        'source_type': ['social', 'video', 'social', 'video', 'social', 'video', 'social', 'video', 'social', 'video'],
        'topics': [
            ['sports', 'entertainment'],
            ['music', 'celebrity'],
            ['sports', 'celebrity'],
            ['music', 'entertainment'],
            ['sports', 'music'],
            ['celebrity', 'entertainment'],
            ['sports', 'celebrity'],
            ['music', 'entertainment'],
            ['entertainment', 'celebrity'],
            ['sports', 'music']
        ]
    })


class TestNumericalDrift:
    """Tests for numerical drift detection."""

    def test_detector_initialization(self):
        """Test that DriftDetector initializes correctly."""
        detector = DriftDetector()
        assert detector is not None
        assert hasattr(detector, 'calculate_numerical_drift')
        assert hasattr(detector, 'DRIFT_THRESHOLD')

    def test_numerical_drift_basic(self, sample_data, new_data):
        """Test basic numerical drift detection."""
        detector = DriftDetector()
        baseline_series = sample_data['word_count']
        new_series = new_data['word_count']
        
        assert len(baseline_series) > 0, "Baseline series is empty"
        assert len(new_series) > 0, "New series is empty"
        
        drift = detector.calculate_numerical_drift(baseline_series, new_series)
        assert drift is not None, "Drift calculation returned None"
        """Test that numerical drift returns expected structure."""
        detector = DriftDetector()
        drift = detector.calculate_numerical_drift(sample_data['word_count'], new_data['word_count'])
        
        assert drift is not None, "Drift calculation returned None for valid data"
        assert isinstance(drift, dict), f"Expected dict, got {type(drift)}"
        assert 'mean_shift' in drift, "Missing 'mean_shift' key"
        assert 'std_ratio' in drift, "Missing 'std_ratio' key"
        assert 'ks_statistic' in drift, "Missing 'ks_statistic' key"
        assert 'ks_pvalue' in drift, "Missing 'ks_pvalue' key"
        assert 'baseline_mean' in drift, "Missing 'baseline_mean' key"
        assert 'new_mean' in drift, "Missing 'new_mean' key"
        assert 'baseline_std' in drift, "Missing 'baseline_std' key"
        assert 'new_std' in drift, "Missing 'new_std' key"
        assert 'has_drift' in drift, "Missing 'has_drift' key"
        assert 'severity' in drift, "Missing 'severity' key"
        assert 'percent_mean_change' in drift, "Missing 'percent_mean_change' key"

    
    def test_numerical_drift_debug(self, sample_data, new_data):
        """Debug test to print drift information."""
        detector = DriftDetector()
        drift = detector.calculate_numerical_drift(sample_data['word_count'], new_data['word_count'])
        
        print("\n" + "="*60)
        print("NUMERICAL DRIFT DEBUG INFO")
        print("="*60)
        print(f"Drift Result: {drift}")
        if drift:
            for key, value in drift.items():
                print(f"  {key}: {value} (type: {type(value).__name__})")
        print("="*60)

    def test_numerical_drift_high_drift(self, sample_data, high_drift_data):
        """Test numerical drift detection with high drift."""
        detector = DriftDetector()
        drift = detector.calculate_numerical_drift(sample_data['word_count'], high_drift_data['word_count'])
        
        assert drift['has_drift'] == True
        assert drift['severity'] in ['HIGH', 'MEDIUM']

    def test_numerical_drift_no_drift(self, sample_data):
        """Test numerical drift when data is identical."""
        detector = DriftDetector()
        drift = detector.calculate_numerical_drift(sample_data['word_count'], sample_data['word_count'])
        
        assert drift['mean_shift'] == 0.0

    def test_numerical_drift_with_nan(self, sample_data, new_data):
        """Test numerical drift with NaN values."""
        sample_with_nan = sample_data['word_count'].copy()
        sample_with_nan.iloc[0] = None
        
        detector = DriftDetector()
        drift = detector.calculate_numerical_drift(sample_with_nan, new_data['word_count'])
        
        assert drift is not None


class TestCategoricalDrift:
    """Tests for categorical drift detection."""

    def test_categorical_drift_structure(self, sample_data, new_data):
        """Test that categorical drift returns expected structure."""
        detector = DriftDetector()
        drift = detector.calculate_categorical_drift(sample_data['country'], new_data['country'])
        
        assert drift is not None
        assert 'new_categories' in drift
        assert 'missing_categories' in drift
        assert 'has_drift' in drift
        assert 'total_categories_baseline' in drift
        assert 'total_categories_new' in drift
        assert 'top_categories_baseline' in drift
        assert 'top_categories_new' in drift

    def test_categorical_drift_new_categories(self, sample_data, new_data):
        """Test detection of new categories."""
        detector = DriftDetector()
        drift = detector.calculate_categorical_drift(sample_data['country'], new_data['country'])
        
        new_cats = drift['new_categories']
        assert 'Mexico' in new_cats

    def test_categorical_drift_missing_categories(self, sample_data, new_data):
        """Test detection of missing categories."""
        detector = DriftDetector()
        drift = detector.calculate_categorical_drift(sample_data['country'], new_data['country'])
        
        # Check if any categories are missing
        assert isinstance(drift['missing_categories'], list)

    def test_categorical_drift_no_drift(self, sample_data):
        """Test categorical drift when data is identical."""
        detector = DriftDetector()
        drift = detector.calculate_categorical_drift(sample_data['country'], sample_data['country'])
        
        assert drift['new_categories'] == []
        assert drift['missing_categories'] == []

    def test_categorical_drift_with_nan(self, sample_data, new_data):
        """Test categorical drift with NaN values."""
        sample_with_nan = sample_data['country'].copy()
        sample_with_nan.iloc[0] = None
        
        detector = DriftDetector()
        drift = detector.calculate_categorical_drift(sample_with_nan, new_data['country'])
        
        assert drift is not None


class TestTopicsDrift:
    """Tests for topics drift detection."""

    def test_topics_drift_structure(self, sample_data, new_data):
        """Test that topics drift returns expected structure."""
        detector = DriftDetector()
        drift = detector.calculate_topics_drift(sample_data, new_data)
        
        assert drift is not None
        assert 'new_topics' in drift
        assert 'missing_topics' in drift
        assert 'has_drift' in drift
        assert 'total_topics_baseline' in drift
        assert 'total_topics_new' in drift
        assert 'top_topics_baseline' in drift
        assert 'top_topics_new' in drift

    def test_topics_drift_new_topics(self, sample_data, new_data):
        """Test detection of new topics."""
        detector = DriftDetector()
        drift = detector.calculate_topics_drift(sample_data, new_data)
        
        new_topics = drift['new_topics']
        assert len(new_topics) > 0

    def test_topics_drift_empty_topics(self):
        """Test topics drift with empty topics."""
        detector = DriftDetector()
        empty_df = pd.DataFrame({
            'topics': [[], [], []]
        })
        
        drift = detector.calculate_topics_drift(empty_df, empty_df)
        
        assert drift['has_drift'] == False
        assert len(drift['top_topics_baseline']) == 0

    def test_topics_drift_no_drift(self, sample_data):
        """Test topics drift when data is identical."""
        detector = DriftDetector()
        drift = detector.calculate_topics_drift(sample_data, sample_data)
        
        assert drift['new_topics'] == []
        assert drift['missing_topics'] == []


class TestDetectAllDrift:
    """Tests for detect_all_drift comprehensive method."""

    def test_detect_all_drift_structure(self, sample_data, new_data):
        """Test that detect_all_drift returns all expected features."""
        detector = DriftDetector()
        all_drift = detector.detect_all_drift(sample_data, new_data)
        
        assert 'word_count' in all_drift
        assert 'token_count' in all_drift
        assert 'country' in all_drift
        assert 'source_type' in all_drift
        assert 'topics' in all_drift

    def test_detect_all_drift_completeness(self, sample_data, new_data):
        """Test that all features have drift information."""
        detector = DriftDetector()
        all_drift = detector.detect_all_drift(sample_data, new_data)
        
        for feature in all_drift.values():
            if feature:
                assert 'has_drift' in feature

    def test_detect_all_drift_returns_dict(self, sample_data, new_data):
        """Test that detect_all_drift returns a dictionary."""
        detector = DriftDetector()
        all_drift = detector.detect_all_drift(sample_data, new_data)
        
        assert isinstance(all_drift, dict)

    def test_detect_all_drift_with_missing_columns(self):
        """Test detect_all_drift with missing columns."""
        detector = DriftDetector()
        
        baseline = pd.DataFrame({
            'word_count': [100, 200, 150],
            'country': ['USA', 'USA', 'Canada']
        })
        
        new_data = pd.DataFrame({
            'word_count': [110, 210, 140],
            'country': ['USA', 'Canada', 'USA']
        })
        
        all_drift = detector.detect_all_drift(baseline, new_data)
        
        assert 'word_count' in all_drift
        assert 'country' in all_drift
        assert 'topics' not in all_drift


class TestDatetimeNormalization:
    """Tests for datetime normalization."""

    def test_normalize_datetimes_removes_timezone(self):
        """Test that normalization removes timezone info."""
        detector = DriftDetector()
        
        df = pd.DataFrame({
            'date': pd.date_range(start='2025-01-01', periods=3, tz='UTC')
        })
        
        normalized = detector._normalize_datetimes(df)
        
        assert normalized['date'].dt.tz is None

    def test_normalize_datetimes_preserves_values(self):
        """Test that normalization preserves date values."""
        detector = DriftDetector()
        
        df = pd.DataFrame({
            'date': pd.date_range(start='2025-01-01', periods=3)
        })
        
        normalized = detector._normalize_datetimes(df)
        
        assert len(normalized) == len(df)
        assert normalized['date'].notna().all()


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_none_inputs(self):
        """Test handling of None inputs."""
        detector = DriftDetector()
        
        result = detector.calculate_numerical_drift(None, None)
        assert result is None

    def test_empty_series(self):
        """Test handling of empty series."""
        detector = DriftDetector()
        
        empty = pd.Series([], dtype=float)
        non_empty = pd.Series([1, 2, 3])
        
        result = detector.calculate_numerical_drift(empty, non_empty)
        assert result is None

    def test_all_nan_values(self):
        """Test handling of all NaN values."""
        detector = DriftDetector()
        
        all_nan = pd.Series([None, None, None])
        normal = pd.Series([1, 2, 3])
        
        result = detector.calculate_numerical_drift(all_nan, normal)
        assert result is None

    def test_single_value_series(self):
        """Test handling of single value series."""
        detector = DriftDetector()
        
        single = pd.Series([100])
        normal = pd.Series([1, 2, 3])
        
        result = detector.calculate_numerical_drift(single, normal)
        assert result is not None


class TestChi2SafeTest:
    """Tests for safe chi-square testing."""

    def test_safe_chi2_with_different_categories(self):
        """Test safe chi2 test with different categories."""
        detector = DriftDetector()
        
        baseline = pd.Series(['A', 'A', 'B', 'B', 'C']).value_counts()
        new = pd.Series(['A', 'A', 'B', 'D', 'D']).value_counts()
        
        chi2, p_value = detector._safe_chi2_test(baseline, new)
        
        assert chi2 is not None
        assert p_value is not None

    def test_safe_chi2_identical_distributions(self):
        """Test safe chi2 test with identical distributions."""
        detector = DriftDetector()
        
        data = pd.Series(['A', 'A', 'B', 'B']).value_counts()
        
        chi2, p_value = detector._safe_chi2_test(data, data)
        
        assert chi2 == 0.0 or chi2 < 0.001


if __name__ == '__main__':
    pytest.main([__file__, '-v'])