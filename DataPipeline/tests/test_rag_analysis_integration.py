"""
Integration tests for the complete data pipeline - TFDV style.

Tests the complete workflow: baseline analysis, new data analysis,
and all components working together.
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

# Add the analysis module to the Python path
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), '../scripts/RAG/analysis')
)

from main import DataQualityAnalyzer
from validator import DataValidator
from anomalies_and_bias_detection import AnomalyDetector
from drift import DriftDetector


@pytest.fixture
def sample_baseline_data() -> pd.DataFrame:
    """Create sample baseline data for testing."""
    current_time = datetime.now()
    data = []
    
    for i in range(20):
        data.append({
            "title": f"Sample Title {i}",
            "text": f"This is sample text content number {i} " * 10,
            "link": f"http://example.com/{i}",
            "word_count": 150 + i * 10,
            "token_count": 180 + i * 12,
            "authors": [f"Author{i}"],
            "publish_date": current_time.strftime("%Y-%m-%d"),
            "country": "US" if i % 3 == 0 else "UK",
            "topics": ["tech", "ai"] if i % 2 == 0 else ["science"],
            "source_type": "news" if i % 2 == 0 else "blog"
        })
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_new_data() -> pd.DataFrame:
    """Create sample new data with some drift."""
    current_time = datetime.now()
    data = []
    
    for i in range(10):
        data.append({
            "title": f"New Title {i}",
            "text": f"This is new text content number {i} " * 12,
            "link": f"http://newsite.com/{i}",
            "word_count": 200 + i * 15,
            "token_count": 240 + i * 18,
            "authors": [f"NewAuthor{i}"],
            "publish_date": current_time.strftime("%Y-%m-%d"),
            "country": "US" if i % 2 == 0 else "CA",  # CA is new
            "topics": ["tech", "ml"] if i % 2 == 0 else ["health"],
            "source_type": "news"
        })
    
    return pd.DataFrame(data)


class TestDataQualityAnalyzer:
    """Test suite for DataQualityAnalyzer."""

    def test_load_jsonl(
        self, sample_baseline_data, tmp_path
    ):
        """Test JSONL loading."""
        baseline_file = tmp_path / "baseline.jsonl"
        sample_baseline_data.to_json(
            baseline_file,
            orient="records",
            lines=True
        )

        analyzer = DataQualityAnalyzer()
        df = analyzer.load_jsonl(str(baseline_file))

        assert len(df) == len(sample_baseline_data)
        assert "link" in df.columns
        assert "text" in df.columns

    def test_split_data(self, sample_baseline_data):
        """Test data splitting."""
        analyzer = DataQualityAnalyzer()
        train_df, val_df = analyzer.split_data(sample_baseline_data)

        assert len(train_df) > 0
        assert len(val_df) > 0
        assert len(train_df) + len(val_df) == len(sample_baseline_data)

    def test_infer_schema(self, sample_baseline_data):
        """Test schema inference."""
        analyzer = DataQualityAnalyzer()
        schema = analyzer._infer_schema(sample_baseline_data)

        assert isinstance(schema, dict)
        assert "word_count" in schema
        assert schema["word_count"]["type"] == "int"
        assert "text" in schema
        assert schema["text"]["type"] == "str"
        assert "topics" in schema
        assert schema["topics"]["type"] == "list"

    def test_baseline_analysis(
        self, sample_baseline_data, tmp_path, monkeypatch
    ):
        """Test baseline analysis."""
        monkeypatch.chdir(tmp_path)

        baseline_file = tmp_path / "baseline.jsonl"
        sample_baseline_data.to_json(
            baseline_file,
            orient="records",
            lines=True
        )

        analyzer = DataQualityAnalyzer(is_baseline=True)
        analyzer.BASELINE_FILE = str(baseline_file)
        analyzer.BASELINE_STATS_FILE = str(tmp_path / "baseline_stats.json")

        # Should complete without errors
        analyzer.analyze_baseline()

        # Check that baseline stats were saved
        assert os.path.exists(analyzer.BASELINE_STATS_FILE)

        # Load and verify stats
        with open(analyzer.BASELINE_STATS_FILE, 'r') as f:
            saved_stats = json.load(f)

        assert 'stats' in saved_stats
        assert 'schema' in saved_stats
        assert len(saved_stats['stats']) > 0

    def test_new_data_analysis(
        self, sample_baseline_data, sample_new_data, tmp_path, monkeypatch
    ):
        """Test new data analysis."""
        monkeypatch.chdir(tmp_path)

        baseline_file = tmp_path / "baseline.jsonl"
        new_file = tmp_path / "new.jsonl"

        sample_baseline_data.to_json(
            baseline_file,
            orient="records",
            lines=True
        )
        sample_new_data.to_json(
            new_file,
            orient="records",
            lines=True
        )

        # First run baseline
        analyzer_baseline = DataQualityAnalyzer(is_baseline=True)
        analyzer_baseline.BASELINE_FILE = str(baseline_file)
        analyzer_baseline.BASELINE_STATS_FILE = str(
            tmp_path / "baseline_stats.json"
        )
        analyzer_baseline.analyze_baseline()

        # Then run new data analysis
        analyzer_new = DataQualityAnalyzer(is_baseline=False)
        analyzer_new.BASELINE_FILE = str(baseline_file)
        analyzer_new.NEW_DATA_FILE = str(new_file)
        analyzer_new.BASELINE_STATS_FILE = str(
            tmp_path / "baseline_stats.json"
        )

        # Should complete without errors
        analyzer_new.analyze_new_data()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])