"""
Unit tests for the AnomalyDetector class.

Tests anomaly detection and bias detection functionality with various
data scenarios including edge cases and typical use cases.
"""

import sys
import os
from typing import Dict

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
def sample_df() -> pd.DataFrame:
    """Create a sample DataFrame for testing.
    
    Returns:
        pd.DataFrame: Sample data with various data quality issues.
    """
    return pd.DataFrame([
        {
            "link": "https://example.com/1",
            "text": "Short text",
            "word_count": 10,
            "topics": [],
            "country": "US",
            "source_type": "news",
            "publish_date": "2025-09-01",
        },
        {
            "link": "https://example.com/2",
            "text": "This is a valid text content with sufficient words " * 5,
            "word_count": 50,
            "topics": ["science", "tech"],
            "country": "US",
            "source_type": "blog",
            "publish_date": "2025-10-10",
        },
        {
            "link": "https://example.com/3",
            "text": "",
            "word_count": 0,
            "topics": None,
            "country": "UK",
            "source_type": "news",
            "publish_date": None,
        },
    ])


@pytest.fixture
def large_sample_df() -> pd.DataFrame:
    """Create a larger sample DataFrame for bias testing.
    
    Returns:
        pd.DataFrame: Larger dataset with intentional biases.
    """
    base_data = {
        "link": [f"https://example.com/{i}" for i in range(100)],
        "text": ["Valid text content " * 10 for _ in range(100)],
        "word_count": np.random.randint(150, 450, 100),
        "topics": [["tech", "ai"] for _ in range(80)] + 
                  [["sports"] for _ in range(20)],
        "country": ["US"] * 85 + ["UK"] * 10 + ["CA"] * 5,
        "source_type": ["news"] * 90 + ["blog"] * 10,
        "publish_date": pd.date_range(
            start='2025-01-01', 
            periods=100, 
            freq='D'
        ).astype(str).tolist(),
    }
    return pd.DataFrame(base_data)


class TestAnomalyDetection:
    """Test suite for anomaly detection functionality."""
    
    def test_detect_all_anomalies_basic(
        self, 
        sample_df: pd.DataFrame
    ) -> None:
        """Test basic anomaly detection functionality.
        
        Args:
            sample_df: Sample DataFrame fixture.
        """
        detector = AnomalyDetector()
        results = detector.detect_all_anomalies(sample_df, "TestDataset")
        
        assert isinstance(results, dict)
        assert "text_anomalies" in results
        assert isinstance(results["text_anomalies"], list)
        assert results["total_anomalies"] == len(results["text_anomalies"])
        
        # Check that we detected the expected anomaly types
        anomaly_types = {
            anomaly["expectation"] 
            for anomaly in results["text_anomalies"]
        }
        
        expected_types = {
            "word_count_out_of_bounds",
            "text_null_or_blank",
            "topics_not_list_or_null",
        }
        
        # At least one expected type should be found
        assert len(anomaly_types & expected_types) > 0
    
    def test_word_count_anomalies(
        self, 
        sample_df: pd.DataFrame
    ) -> None:
        """Test word count boundary detection.
        
        Args:
            sample_df: Sample DataFrame fixture.
        """
        detector = AnomalyDetector()
        results = detector.detect_all_anomalies(sample_df, "TestDataset")
        
        word_count_anomalies = [
            anomaly for anomaly in results["text_anomalies"]
            if anomaly["column"] == "word_count"
        ]
        
        # We should detect the entries with word counts 10 and 0
        assert len(word_count_anomalies) >= 2
    
    def test_text_anomalies(
        self, 
        sample_df: pd.DataFrame
    ) -> None:
        """Test text content anomaly detection.
        
        Args:
            sample_df: Sample DataFrame fixture.
        """
        detector = AnomalyDetector()
        results = detector.detect_all_anomalies(sample_df, "TestDataset")
        
        text_anomalies = [
            anomaly for anomaly in results["text_anomalies"]
            if anomaly["column"] == "text"
        ]
        
        # Should detect the blank text in row 3
        assert len(text_anomalies) >= 1
    
    def test_topics_anomalies(
        self, 
        sample_df: pd.DataFrame
    ) -> None:
        """Test topics field anomaly detection.
        
        Args:
            sample_df: Sample DataFrame fixture.
        """
        detector = AnomalyDetector()
        results = detector.detect_all_anomalies(sample_df, "TestDataset")
        
        topics_anomalies = [
            anomaly for anomaly in results["text_anomalies"]
            if anomaly["column"] == "topics"
        ]
        
        # Should detect the None value in row 3
        assert len(topics_anomalies) >= 1
    
    def test_empty_dataframe(self) -> None:
        """Test handling of empty DataFrame."""
        detector = AnomalyDetector()
        empty_df = pd.DataFrame()
        
        results = detector.detect_all_anomalies(
            empty_df, 
            "EmptyDataset"
        )
        
        assert results["total_anomalies"] == 0
        assert len(results["text_anomalies"]) == 0
    
    def test_missing_columns(self) -> None:
        """Test handling of DataFrames with missing expected columns."""
        detector = AnomalyDetector()
        partial_df = pd.DataFrame({
            "link": ["https://example.com/1"],
            "other_column": ["some_value"]
        })
        
        results = detector.detect_all_anomalies(
            partial_df, 
            "PartialDataset"
        )
        
        # Should handle gracefully without errors
        assert isinstance(results, dict)
        assert "total_anomalies" in results


class TestBiasDetection:
    """Test suite for bias detection functionality."""
    
    def test_detect_bias_country(
        self, 
        large_sample_df: pd.DataFrame
    ) -> None:
        """Test country bias detection.
        
        Args:
            large_sample_df: Large sample DataFrame fixture.
        """
        detector = AnomalyDetector()
        bias = detector.detect_bias(large_sample_df)
        
        assert isinstance(bias, dict)
        assert "country_bias" in bias
        assert "dominant" in bias["country_bias"]
        assert "percentage" in bias["country_bias"]
        
        # US should be dominant (85% of data)
        assert bias["country_bias"]["dominant"] == "US"
        assert bias["country_bias"]["percentage"] > 80
    
    def test_detect_bias_source(
        self, 
        large_sample_df: pd.DataFrame
    ) -> None:
        """Test source type bias detection.
        
        Args:
            large_sample_df: Large sample DataFrame fixture.
        """
        detector = AnomalyDetector()
        bias = detector.detect_bias(large_sample_df)
        
        assert "source_bias" in bias
        assert bias["source_bias"]["dominant"] == "news"
        assert bias["source_bias"]["percentage"] == 90.0
    
    def test_detect_bias_topics(
        self, 
        large_sample_df: pd.DataFrame
    ) -> None:
        """Test topics bias detection.
        
        Args:
            large_sample_df: Large sample DataFrame fixture.
        """
        detector = AnomalyDetector()
        bias = detector.detect_bias(large_sample_df)
        
        assert isinstance(bias, dict)
        if "top_topics" in bias:
            assert isinstance(bias["top_topics"], dict)
            # Should have tech and ai as top topics
            top_topics = list(bias["top_topics"].keys())
            assert "tech" in top_topics or "ai" in top_topics
    
    def test_detect_bias_temporal(
        self, 
        sample_df: pd.DataFrame
    ) -> None:
        """Test temporal distribution analysis.
        
        Args:
            sample_df: Sample DataFrame fixture.
        """
        detector = AnomalyDetector()
        bias = detector.detect_bias(sample_df)
        
        assert isinstance(bias, dict)
        if "temporal_distribution" in bias:
            td = bias["temporal_distribution"]
            assert "last_30_days" in td
            assert "last_90_days" in td
            assert "last_year" in td
            assert "older" in td
            assert "total_with_dates" in td
    
    def test_bias_with_null_values(self) -> None:
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
    
    def test_remove_detected_anomalies(
        self, 
        sample_df: pd.DataFrame
    ) -> None:
        """Test removal of detected anomalies.
        
        Args:
            sample_df: Sample DataFrame fixture.
        """
        detector = AnomalyDetector()
        
        # Test with no anomalies dict
        result_df = detector.remove_detected_anomalies(
            sample_df, 
            {}
        )
        assert len(result_df) == len(sample_df)
        
        # Test with anomalies dict containing rows
        anomalies_dict = {"rows": [0, 2]}
        result_df = detector.remove_detected_anomalies(
            sample_df, 
            anomalies_dict
        )
        assert len(result_df) == len(sample_df) - 2
    
    def test_array_like_topics(self) -> None:
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
        })
        
        detector = AnomalyDetector()
        results = detector.detect_all_anomalies(
            df_with_arrays, 
            "ArrayDataset"
        )
        
        # Should handle numpy arrays without errors
        assert isinstance(results, dict)
        
        # Test bias detection with arrays
        bias = detector.detect_bias(df_with_arrays)
        assert isinstance(bias, dict)
        if "top_topics" in bias:
            # Should extract topics from numpy arrays
            assert len(bias["top_topics"]) > 0
    
    def test_special_characters_in_text(self) -> None:
        """Test handling of special characters in text."""
        df_special = pd.DataFrame({
            "text": [
                "Normal text",
                "Text with 你好世界",
                "Tab\tseparated\ttext",
                "aaaaaaaaaaaaaaaa",
            ],
            "word_count": [100, 200, 180, 50],
        })
        
        detector = AnomalyDetector()
        results = detector.detect_all_anomalies(
            df_special, 
            "SpecialCharDataset"
        )
        
        # Should handle special characters without errors
        assert isinstance(results, dict)
        assert "text_anomalies" in results
    
    def test_very_large_values(self) -> None:
        """Test handling of very large numeric values."""
        df_large = pd.DataFrame({
            "word_count": [
                100, 
                1000, 
                10000, 
                100000, 
                float('inf')
            ],
            "text": ["text"] * 5,
        })
        
        detector = AnomalyDetector()
        results = detector.detect_all_anomalies(
            df_large, 
            "LargeValueDataset"
        )
        
        # Should detect extreme values as anomalies
        word_anomalies = [
            a for a in results["text_anomalies"] 
            if a["column"] == "word_count"
        ]
        assert len(word_anomalies) >= 2  # At least the very large values


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])