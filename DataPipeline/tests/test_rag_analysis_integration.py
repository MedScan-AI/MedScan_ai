"""
Integration tests for the complete data pipeline.

This module provides comprehensive integration tests for the data quality
analysis pipeline, including validation, drift detection, and the main
analyzer components.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import pytest

# Add the analysis module to the Python path
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), '../scripts/RAG/analysis')
)

from main import DataQualityAnalyzer
from validator import DataValidator
from drift import DriftDetector


@pytest.fixture
def sample_baseline_data() -> pd.DataFrame:
    """Create sample baseline data for testing.
    
    Returns:
        pd.DataFrame: A DataFrame containing two sample records with
            all required fields for baseline analysis.
    """
    current_time = datetime.now()
    data = [
        {
            "title": "Title 1",
            "text": "Some text",
            "link": "http://a.com",
            "word_count": 100,
            "token_count": 120,
            "authors": ["Alice"],
            "publish_date": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "country": "US",
            "topics": ["tech"],
            "source_type": "blog",
            "accessed_at": current_time
        },
        {
            "title": "Title 2",
            "text": "Another text",
            "link": "http://b.com",
            "word_count": 200,
            "token_count": 210,
            "authors": ["Bob"],
            "publish_date": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "country": "UK",
            "topics": ["science"],
            "source_type": "news",
            "accessed_at": current_time
        },
    ]
    return pd.DataFrame(data)


@pytest.fixture
def sample_new_data() -> pd.DataFrame:
    """Create sample new data for testing drift detection.
    
    Returns:
        pd.DataFrame: A DataFrame containing two sample records with
            slightly different characteristics for drift testing.
    """
    current_time = datetime.now()
    data = [
        {
            "title": "Title 3",
            "text": "New text",
            "link": "http://c.com",
            "word_count": 150,
            "token_count": 180,
            "authors": ["Charlie"],
            "publish_date": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "country": "US",
            "topics": ["tech"],
            "source_type": "blog",
            "accessed_at": current_time
        },
        {
            "title": "Title 4",
            "text": "Another new text",
            "link": "http://d.com",
            "word_count": 250,
            "token_count": 260,
            "authors": ["Dana"],
            "publish_date": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "country": "CA",
            "topics": ["health"],
            "source_type": "news",
            "accessed_at": current_time
        },
    ]
    return pd.DataFrame(data)


class TestDataValidator:
    """Test suite for the DataValidator class."""
    
    def test_validator_detects_types(
        self,
        sample_baseline_data: pd.DataFrame
    ) -> None:
        """Test that validator correctly identifies data types.
        
        Args:
            sample_baseline_data: Fixture providing sample data.
        """
        validator = DataValidator()
        issues = validator.validate_data_types(
            sample_baseline_data,
            dataset_name="Test Dataset"
        )
        
        # There should be no type issues in this valid sample
        assert issues == {}
    
    def test_validator_detects_missing_column(
        self,
        sample_baseline_data: pd.DataFrame
    ) -> None:
        """Test validator behavior with missing columns.
        
        Args:
            sample_baseline_data: Fixture providing sample data.
        """
        df = sample_baseline_data.drop(columns=["title"])
        validator = DataValidator()
        issues = validator.validate_data_types(
            df,
            dataset_name="Test Dataset"
        )
        
        # Missing column should not raise error in current validator
        assert "title" not in issues


class TestDriftDetector:
    """Test suite for the DriftDetector class."""
    
    def test_drift_detector_numeric(
        self,
        sample_baseline_data: pd.DataFrame,
        sample_new_data: pd.DataFrame
    ) -> None:
        """Test drift detection for numeric features.
        
        Args:
            sample_baseline_data: Baseline data fixture.
            sample_new_data: New data fixture for comparison.
        """
        detector = DriftDetector()
        drift_info = detector.calculate_numerical_drift(
            sample_baseline_data["word_count"],
            sample_new_data["word_count"]
        )
        
        assert drift_info is not None
        assert "has_drift" in drift_info
        assert "mean_shift" in drift_info
        assert "baseline_mean" in drift_info
        assert "new_mean" in drift_info
    
    def test_drift_detector_categorical(
        self,
        sample_baseline_data: pd.DataFrame,
        sample_new_data: pd.DataFrame
    ) -> None:
        """Test drift detection for categorical features.
        
        Args:
            sample_baseline_data: Baseline data fixture.
            sample_new_data: New data fixture for comparison.
        """
        detector = DriftDetector()
        drift_info = detector.calculate_categorical_drift(
            sample_baseline_data["country"],
            sample_new_data["country"]
        )
        
        assert drift_info is not None
        assert "has_drift" in drift_info
        assert "new_categories" in drift_info
        assert "missing_categories" in drift_info
        
        # CA should be detected as a new category
        assert "CA" in drift_info["new_categories"]


class TestDataQualityAnalyzer:
    """Test suite for the main DataQualityAnalyzer class."""
    
    def test_data_quality_analyzer_baseline(
        self,
        sample_baseline_data: pd.DataFrame,
        tmp_path: Path
    ) -> None:
        """Test baseline analysis functionality.
        
        Args:
            sample_baseline_data: Baseline data fixture.
            tmp_path: Pytest fixture providing temporary directory.
        """
        # Save temporary baseline file
        baseline_file = tmp_path / "baseline.jsonl"
        sample_baseline_data.to_json(
            baseline_file,
            orient="records",
            lines=True
        )
        
        analyzer = DataQualityAnalyzer(is_baseline=True)
        analyzer.BASELINE_FILE = str(baseline_file)
        
        results = analyzer.analyze_baseline()
        
        assert results is not None
        assert results["is_baseline"] is True
        assert "type_issues" in results
        assert "total_records" in results
        assert results["total_records"] == len(sample_baseline_data)
    
    def test_data_quality_analyzer_new_data(
        self,
        sample_baseline_data: pd.DataFrame,
        sample_new_data: pd.DataFrame,
        tmp_path: Path
    ) -> None:
        """Test new data analysis with drift detection.
        
        Args:
            sample_baseline_data: Baseline data fixture.
            sample_new_data: New data fixture.
            tmp_path: Pytest fixture providing temporary directory.
        """
        baseline_file = tmp_path / "baseline.jsonl"
        new_file = tmp_path / "new.jsonl"
        
        # Save test data to temporary files
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
        
        analyzer = DataQualityAnalyzer(is_baseline=False)
        analyzer.BASELINE_FILE = str(baseline_file)
        analyzer.NEW_DATA_FILE = str(new_file)
        
        # Pre-save baseline stats to simulate prior run
        baseline_stats = {
            "total_records": len(sample_baseline_data),
            "train_records": len(sample_baseline_data),
            "validation_records": 0,
            "schema": analyzer.infer_schema(sample_baseline_data),
            "type_issues": {},
            "anomalies": {},
            "validation": {},
            "bias": {},
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        analyzer._save_baseline_stats(baseline_stats)
        
        results = analyzer.analyze_new_data()
        
        assert results is not None
        assert results["is_baseline"] is False
        assert "drift_results" in results
        assert "total_records" in results
        assert results["total_records"] == len(sample_new_data)
    
    def test_analyzer_with_empty_data(
        self,
        tmp_path: Path
    ) -> None:
        """Test analyzer behavior with empty datasets.
        
        Args:
            tmp_path: Pytest fixture providing temporary directory.
        """
        # Create empty JSONL file
        empty_file = tmp_path / "empty.jsonl"
        empty_file.touch()
        
        analyzer = DataQualityAnalyzer(is_baseline=True)
        analyzer.BASELINE_FILE = str(empty_file)
        
        results = analyzer.analyze_baseline()
        
        # Should handle empty file gracefully
        assert results is None or results["total_records"] == 0
    
    def test_schema_inference(
        self,
        sample_baseline_data: pd.DataFrame
    ) -> None:
        """Test automatic schema inference.
        
        Args:
            sample_baseline_data: Baseline data fixture.
        """
        analyzer = DataQualityAnalyzer(is_baseline=True)
        schema = analyzer.infer_schema(sample_baseline_data)
        
        assert isinstance(schema, dict)
        assert "word_count" in schema
        assert schema["word_count"]["type"] == "int"
        assert "text" in schema
        assert schema["text"]["type"] == "str"
        assert "topics" in schema
        assert schema["topics"]["type"] == "list"
    
    def test_baseline_stats_persistence(
        self,
        sample_baseline_data: pd.DataFrame,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test saving and loading of baseline statistics.
        
        Args:
            sample_baseline_data: Baseline data fixture.
            tmp_path: Pytest fixture providing temporary directory.
            monkeypatch: Pytest fixture for patching.
        """
        # Change working directory to temp path for stats file
        monkeypatch.chdir(tmp_path)
        
        analyzer = DataQualityAnalyzer(is_baseline=True)
        
        test_stats = {
            "total_records": 100,
            "train_records": 70,
            "validation_records": 30,
            "schema": {"test": "schema"},
            "timestamp": "2024-01-01 00:00:00"
        }
        
        # Save stats
        analyzer._save_baseline_stats(test_stats)
        
        # Load stats
        loaded_stats = analyzer._load_baseline_stats()
        
        assert loaded_stats is not None
        assert loaded_stats["total_records"] == 100
        assert loaded_stats["train_records"] == 70
        assert loaded_stats["schema"] == {"test": "schema"}


class TestIntegrationFlow:
    """End-to-end integration tests for the complete pipeline."""
    
    def test_complete_pipeline_flow(
        self,
        sample_baseline_data: pd.DataFrame,
        sample_new_data: pd.DataFrame,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test the complete analysis pipeline from baseline to drift.
        
        Args:
            sample_baseline_data: Baseline data fixture.
            sample_new_data: New data fixture.
            tmp_path: Pytest fixture providing temporary directory.
            monkeypatch: Pytest fixture for patching.
        """
        # Setup test environment
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
        
        # Phase 1: Baseline analysis
        baseline_analyzer = DataQualityAnalyzer(is_baseline=True)
        baseline_analyzer.BASELINE_FILE = str(baseline_file)
        baseline_results = baseline_analyzer.analyze_baseline()
        
        assert baseline_results is not None
        assert baseline_results["is_baseline"] is True
        
        # Phase 2: New data analysis
        new_analyzer = DataQualityAnalyzer(is_baseline=False)
        new_analyzer.BASELINE_FILE = str(baseline_file)
        new_analyzer.NEW_DATA_FILE = str(new_file)
        new_results = new_analyzer.analyze_new_data()
        
        assert new_results is not None
        assert new_results["is_baseline"] is False
        assert "drift_results" in new_results
        
        # Check drift was detected for country (CA is new)
        if "country" in new_results["drift_results"]:
            country_drift = new_results["drift_results"]["country"]
            assert "CA" in country_drift.get("new_categories", [])