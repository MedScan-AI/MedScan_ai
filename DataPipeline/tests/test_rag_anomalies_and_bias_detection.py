"""
Unit tests for the AnomalyDetector class - TFDV style.

Tests baseline statistics computation, anomaly detection,
completeness checks, and bias detection functionality.
"""

import sys
import os

import pytest
import pandas as pd
import numpy as np

# Add the analysis module to the path
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), '../scripts/RAG/analysis')
)

from anomalies_and_bias_detection import AnomalyDetector


@pytest.fixture
def clean_baseline_df() -> pd.DataFrame:
    """Create a clean baseline DataFrame."""
    return pd.DataFrame([
        {
            "link": "https://example.com/1",
            "title": "Valid Title Here",
            "text": "This is valid text content with sufficient words",
            "word_count": 150,
            "token_count": 180,
            "topics": ["science", "tech"],
            "authors": ["Alice"],
            "country": "US",
            "source_type": "news",
            "publish_date": "2025-09-01",
        },
        {
            "link": "https://example.com/2",
            "title": "Another Valid Title",
            "text": "More valid text content here for testing purposes",
            "word_count": 200,
            "token_count": 240,
            "topics": ["health", "science"],
            "authors": ["Bob", "Carol"],
            "country": "US",
            "source_type": "blog",
            "publish_date": "2025-10-10",
        },
        {
            "link": "https://example.com/3",
            "title": "Third Title Example",
            "text": "Additional text content for baseline statistics",
            "word_count": 180,
            "token_count": 220,
            "topics": ["tech"],
            "authors": ["David"],
            "country": "UK",
            "source_type": "news",
            "publish_date": "2025-10-15",
        },
    ])


@pytest.fixture
def anomalous_df() -> pd.DataFrame:
    """Create DataFrame with anomalies."""
    return pd.DataFrame([
        {
            "link": "https://example.com/10",
            "title": "AI",  # Too short
            "text": "Short",  # Too short
            "word_count": 10,  # Below baseline
            "token_count": 15,  # Below baseline
            "topics": None,  # Null
            "authors": [],
            "country": "US",
            "source_type": "news",
            "publish_date": "2025-09-01",
        },
        {
            "link": "https://example.com/11",
            "title": "Very Long Title " * 100,  # Too long
            "text": "Normal text content here",
            "word_count": 500,  # Above baseline
            "token_count": 600,  # Above baseline
            "topics": ["a", "b", "c"] * 20,  # List too long
            "authors": ["Author"],
            "country": "CA",
            "source_type": "blog",
            "publish_date": "2025-10-10",
        },
    ])


@pytest.fixture
def large_sample_df() -> pd.DataFrame:
    """Create larger DataFrame for bias testing."""
    base_data = {
        "link": [f"https://example.com/{i}" for i in range(100)],
        "title": [f"Title {i}" for i in range(100)],
        "text": ["Valid text content " * 10 for _ in range(100)],
        "word_count": np.random.randint(150, 250, 100),
        "token_count": np.random.randint(180, 300, 100),
        "topics": [["tech", "ai"] for _ in range(80)] +
                  [["sports"] for _ in range(20)],
        "authors": [["Author"] for _ in range(100)],
        "country": ["US"] * 85 + ["UK"] * 10 + ["CA"] * 5,
        "source_type": ["news"] * 90 + ["blog"] * 10,
        "publish_date": pd.date_range(
            start='2025-01-01',
            periods=100,
            freq='D'
        ).astype(str).tolist(),
    }
    return pd.DataFrame(base_data)


class TestComputeBaselineStats:
    """Test suite for compute_baseline_stats."""

    def test_compute_stats_basic(self, clean_baseline_df: pd.DataFrame):
        """Test basic baseline stats computation."""
        detector = AnomalyDetector()
        stats = detector.compute_baseline_stats(clean_baseline_df)

        assert isinstance(stats, dict)
        assert len(stats) > 0

        # Check numeric column stats
        assert "word_count" in stats
        assert stats["word_count"]["type"] == "numeric"
        assert "min" in stats["word_count"]
        assert "max" in stats["word_count"]
        assert "mean" in stats["word_count"]
        assert "std" in stats["word_count"]
        assert "null_pct" in stats["word_count"]

    def test_compute_stats_text_columns(self, clean_baseline_df):
        """Test text column statistics."""
        detector = AnomalyDetector()
        stats = detector.compute_baseline_stats(clean_baseline_df)

        assert "title" in stats
        assert stats["title"]["type"] == "text"
        assert "min_len" in stats["title"]
        assert "max_len" in stats["title"]
        assert "avg_len" in stats["title"]

    def test_compute_stats_list_columns(self, clean_baseline_df):
        """Test list column statistics."""
        detector = AnomalyDetector()
        stats = detector.compute_baseline_stats(clean_baseline_df)

        assert "topics" in stats
        assert stats["topics"]["type"] == "list"
        assert "min_len" in stats["topics"]
        assert "max_len" in stats["topics"]
        assert "avg_len" in stats["topics"]

    def test_compute_stats_empty_dataframe(self):
        """Test with empty DataFrame."""
        detector = AnomalyDetector()
        empty_df = pd.DataFrame()
        stats = detector.compute_baseline_stats(empty_df)

        assert stats == {}

    def test_compute_stats_skips_error_column(self):
        """Test that error column is skipped."""
        df = pd.DataFrame({
            "error": ["error1", "error2"],
            "word_count": [100, 200]
        })
        detector = AnomalyDetector()
        stats = detector.compute_baseline_stats(df)

        assert "error" not in stats
        assert "word_count" in stats


class TestDetectAnomalousRecords:
    """Test suite for detect_anomalous_records."""

    def test_detect_no_anomalies(self, clean_baseline_df):
        """Test detection when no anomalies exist."""
        detector = AnomalyDetector()
        baseline_stats = detector.compute_baseline_stats(clean_baseline_df)
        anomalies = detector.detect_anomalous_records(
            clean_baseline_df, baseline_stats, "Test"
        )

        assert isinstance(anomalies, list)
        assert len(anomalies) == 0

    def test_detect_word_count_anomalies(
        self, clean_baseline_df, anomalous_df
    ):
        """Test detection of word_count anomalies."""
        detector = AnomalyDetector()
        baseline_stats = detector.compute_baseline_stats(clean_baseline_df)
        anomalies = detector.detect_anomalous_records(
            anomalous_df, baseline_stats, "Test"
        )

        assert len(anomalies) > 0

        # Check that word_count violations are detected
        violations = []
        for record in anomalies:
            violations.extend(record['violations'])

        word_count_violations = [
            v for v in violations if 'word_count' in v
        ]
        assert len(word_count_violations) > 0

    def test_detect_text_length_anomalies(
        self, clean_baseline_df, anomalous_df
    ):
        """Test detection of text length anomalies."""
        detector = AnomalyDetector()
        baseline_stats = detector.compute_baseline_stats(clean_baseline_df)
        anomalies = detector.detect_anomalous_records(
            anomalous_df, baseline_stats, "Test"
        )

        # Should detect short text and long title
        assert len(anomalies) > 0

    def test_detect_null_violations(self, clean_baseline_df):
        """Test detection of unexpected nulls."""
        # Create data with null in low-null-rate column
        df_with_nulls = pd.DataFrame({
            "link": ["https://example.com/1"],
            "title": [None],  # Null in title
            "text": ["Content"],
            "word_count": [150],
            "token_count": [180],
            "topics": [["tech"]],
            "authors": [["Author"]],
            "country": ["US"],
            "source_type": ["news"]
        })

        detector = AnomalyDetector()
        baseline_stats = detector.compute_baseline_stats(clean_baseline_df)
        anomalies = detector.detect_anomalous_records(
            df_with_nulls, baseline_stats, "Test"
        )

        # Should detect null title if baseline has low null rate
        violations = []
        for record in anomalies:
            violations.extend(record['violations'])

        title_violations = [v for v in violations if 'title' in v]
        assert len(title_violations) > 0

    def test_anomaly_record_structure(
        self, clean_baseline_df, anomalous_df
    ):
        """Test that anomalous records have correct structure."""
        detector = AnomalyDetector()
        baseline_stats = detector.compute_baseline_stats(clean_baseline_df)
        anomalies = detector.detect_anomalous_records(
            anomalous_df, baseline_stats, "Test"
        )

        if anomalies:
            record = anomalies[0]
            assert 'index' in record
            assert 'link' in record
            assert 'violations' in record
            assert isinstance(record['violations'], list)


class TestDetectCompletenessIssues:
    """Test suite for detect_completeness_issues."""

    def test_detect_missing_topics(self):
        """Test detection of missing topics."""
        df = pd.DataFrame({
            "link": ["https://example.com/1", "https://example.com/2"],
            "title": ["Title 1", "Title 2"],
            "text": ["Text 1", "Text 2"],
            "topics": [None, ["tech"]]
        })

        detector = AnomalyDetector()
        issues = detector.detect_completeness_issues(df, "Test")

        assert "MISSING_TOPICS" in issues
        assert len(issues["MISSING_TOPICS"]) == 1
        assert issues["MISSING_TOPICS"][0]['index'] == 0

    def test_detect_empty_text(self):
        """Test detection of empty text."""
        df = pd.DataFrame({
            "link": ["https://example.com/1", "https://example.com/2"],
            "title": ["Title 1", "Title 2"],
            "text": [None, "Valid text"],
            "topics": [["tech"], ["science"]]
        })

        detector = AnomalyDetector()
        issues = detector.detect_completeness_issues(df, "Test")

        assert "EMPTY_TEXT" in issues
        assert len(issues["EMPTY_TEXT"]) == 1

    def test_detect_missing_title(self):
        """Test detection of missing title."""
        df = pd.DataFrame({
            "link": ["https://example.com/1"],
            "title": [""],
            "text": ["Content"],
            "topics": [["tech"]]
        })

        detector = AnomalyDetector()
        issues = detector.detect_completeness_issues(df, "Test")

        assert "MISSING_TITLE" in issues
        assert len(issues["MISSING_TITLE"]) == 1

    def test_detect_empty_topics_list(self):
        """Test detection of empty topics list."""
        df = pd.DataFrame({
            "link": ["https://example.com/1"],
            "title": ["Title"],
            "text": ["Content"],
            "topics": [[]]
        })

        detector = AnomalyDetector()
        issues = detector.detect_completeness_issues(df, "Test")

        assert "MISSING_TOPICS" in issues
        assert len(issues["MISSING_TOPICS"]) == 1

    def test_no_completeness_issues(self, clean_baseline_df):
        """Test when no completeness issues exist."""
        detector = AnomalyDetector()
        issues = detector.detect_completeness_issues(
            clean_baseline_df, "Test"
        )

        # Clean data should have no issues
        assert len(issues) == 0


class TestBiasDetection:
    """Test suite for bias detection functionality."""

    def test_detect_bias_country(self, large_sample_df):
        """Test country bias detection."""
        detector = AnomalyDetector()
        bias = detector.detect_bias(large_sample_df)

        assert isinstance(bias, dict)
        assert "country_bias" in bias
        assert "dominant" in bias["country_bias"]
        assert "percentage" in bias["country_bias"]

        # US should be dominant (85% of data)
        assert bias["country_bias"]["dominant"] == "US"
        assert bias["country_bias"]["percentage"] > 80

    def test_detect_bias_source(self, large_sample_df):
        """Test source type bias detection."""
        detector = AnomalyDetector()
        bias = detector.detect_bias(large_sample_df)

        assert "source_bias" in bias
        assert bias["source_bias"]["dominant"] == "news"
        assert bias["source_bias"]["percentage"] == 90.0

    def test_detect_bias_topics(self, large_sample_df):
        """Test topics bias detection."""
        detector = AnomalyDetector()
        bias = detector.detect_bias(large_sample_df)

        assert isinstance(bias, dict)
        if "top_topics" in bias:
            assert isinstance(bias["top_topics"], dict)
            # Should have tech and ai as top topics
            top_topics = list(bias["top_topics"].keys())
            assert "tech" in top_topics or "ai" in top_topics

    def test_detect_bias_temporal(self, clean_baseline_df):
        """Test temporal distribution analysis."""
        detector = AnomalyDetector()
        bias = detector.detect_bias(clean_baseline_df)

        assert isinstance(bias, dict)
        # Temporal may not be present if dates fail to parse
        if "temporal_distribution" in bias:
            td = bias["temporal_distribution"]
            assert "last_30_days" in td
            assert "last_90_days" in td
            assert "last_year" in td
            assert "older" in td
            assert "total_with_dates" in td

    def test_bias_with_null_values(self):
        """Test bias detection with null values."""
        df_with_nulls = pd.DataFrame({
            "country": ["US", "US", None, "UK", None],
            "source_type": ["news", None, "blog", None, "news"],
            "topics": [["tech"], None, [], ["sports"], None],
            "publish_date": ["2025-01-01", None, "2025-02-01", None, None]
        })

        detector = AnomalyDetector()
        bias = detector.detect_bias(df_with_nulls)

        # Should handle nulls gracefully
        assert isinstance(bias, dict)
        if "country_bias" in bias:
            assert bias["country_bias"]["dominant"] == "US"


class TestEdgeCases:
    """Test suite for edge cases and error handling."""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        detector = AnomalyDetector()
        empty_df = pd.DataFrame()

        stats = detector.compute_baseline_stats(empty_df)
        assert stats == {}

    def test_missing_columns(self, clean_baseline_df):
        """Test handling of DataFrames with missing columns."""
        detector = AnomalyDetector()

        # Create partial DataFrame
        partial_df = pd.DataFrame({
            "link": ["https://example.com/1"],
            "other_column": ["some_value"]
        })

        baseline_stats = detector.compute_baseline_stats(clean_baseline_df)
        anomalies = detector.detect_anomalous_records(
            partial_df, baseline_stats, "Test"
        )

        # Should handle gracefully without errors
        assert isinstance(anomalies, list)

    def test_array_like_topics(self):
        """Test handling of numpy arrays in topics field."""
        df_with_arrays = pd.DataFrame({
            "topics": [
                np.array(["tech", "ai"]),
                np.array([]),
                None,
                ["science"],
            ],
            "word_count": [200, 300, 400, 500],
            "text": ["text"] * 4,
            "title": ["title"] * 4,
            "link": ["https://example.com/1"] * 4
        })

        detector = AnomalyDetector()

        # Should handle numpy arrays without errors
        stats = detector.compute_baseline_stats(df_with_arrays)
        assert isinstance(stats, dict)

        # Test bias detection with arrays
        bias = detector.detect_bias(df_with_arrays)
        assert isinstance(bias, dict)

    def test_special_characters_in_text(self):
        """Test handling of special characters in text."""
        df_special = pd.DataFrame({
            "link": ["https://example.com/1"] * 4,
            "title": ["Title"] * 4,
            "text": [
                "Normal text",
                "Text with 你好世界",
                "Tab\tseparated\ttext",
                "Emoji 😀 text",
            ],
            "word_count": [100, 200, 180, 150],
            "topics": [["tech"]] * 4
        })

        detector = AnomalyDetector()
        stats = detector.compute_baseline_stats(df_special)

        # Should handle special characters without errors
        assert isinstance(stats, dict)
        assert "text" in stats

    def test_very_large_values(self):
        """Test handling of very large numeric values."""
        df_large = pd.DataFrame({
            "link": ["https://example.com/1"] * 3,
            "title": ["Title"] * 3,
            "text": ["text"] * 3,
            "word_count": [100, 1000, 100000],
            "topics": [["tech"]] * 3
        })

        detector = AnomalyDetector()
        stats = detector.compute_baseline_stats(df_large)

        assert stats["word_count"]["max"] == 100000.0


class TestIntegration:
    """Integration tests for the complete workflow."""

    def test_full_workflow(self, clean_baseline_df, anomalous_df):
        """Test complete workflow: stats -> detect -> report."""
        detector = AnomalyDetector()

        # Step 1: Compute baseline stats
        baseline_stats = detector.compute_baseline_stats(clean_baseline_df)
        assert len(baseline_stats) > 0

        # Step 2: Detect anomalies
        anomalies = detector.detect_anomalous_records(
            anomalous_df, baseline_stats, "Test"
        )
        assert len(anomalies) > 0

        # Step 3: Detect completeness
        completeness = detector.detect_completeness_issues(
            anomalous_df, "Test"
        )
        assert len(completeness) > 0

        # Step 4: Detect bias
        bias = detector.detect_bias(clean_baseline_df)
        assert isinstance(bias, dict)

    def test_hardcoded_word_count_threshold(self, clean_baseline_df):
        """Test that hardcoded word_count threshold is applied."""
        detector = AnomalyDetector()
        baseline_stats = detector.compute_baseline_stats(clean_baseline_df)

        # Create record with word_count below hardcoded threshold
        df_low = pd.DataFrame({
            "link": ["https://example.com/1"],
            "title": ["Title"],
            "text": ["Text"],
            "word_count": [50],  # Below 100
            "token_count": [60],
            "topics": [["tech"]]
        })

        anomalies = detector.detect_anomalous_records(
            df_low, baseline_stats, "Test"
        )

        assert len(anomalies) > 0

        # Check for word_count violation
        violations = anomalies[0]['violations']
        word_violations = [v for v in violations if 'word_count' in v]
        assert len(word_violations) > 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])