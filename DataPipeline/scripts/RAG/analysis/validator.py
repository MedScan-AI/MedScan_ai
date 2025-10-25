"""Data validation module using basic statistical validation."""

import logging
from datetime import datetime
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates data types and schemas."""

    # Configuration
    MIN_WORD_COUNT = 100
    MAX_WORD_COUNT = 50000
    FEATURES_TO_REMOVE = {"error", "url"}
    OPTIONAL_FIELDS = ["country", "publish_date", "authors", "topics"]

    EXPECTED_TYPES = {
        "link": str,
        "title": str,
        "text": str,
        "markdown_title": str,
        "markdown_text": str,
        "source_type": str,
        "accessed_at": "datetime_or_string",
        "word_count": int,
        "token_count": int,
        "authors": list,
        "publish_date": "datetime_or_string",
        "country": str,
        "topics": list,
    }


    def __init__(self, context_root_dir: str = "./gx"):
        """Initialize the validator.

        Args:
            context_root_dir: Root directory (kept for backward compatibility).
        """
        self.baseline_validation_result = None
        self.context_root_dir = context_root_dir



    def validate_data_types(
        self, df: pd.DataFrame, dataset_name: str, schema: Optional[Dict] = None
    ) -> Dict:
        """Validate data types match expectations.

        Args:
            df: DataFrame to validate.
            dataset_name: Name of the dataset for logging.
            schema: Optional schema to use instead of EXPECTED_TYPES.

        Returns:
            Dictionary mapping column names to issue descriptions.
        """
        if schema is None:
            schema = self.EXPECTED_TYPES
        issues = {}

        for column, expected_type in schema.items():
            if column not in df.columns:
                continue

            series = df[column]

            try:
                if expected_type == "datetime":
                    parsed = pd.to_datetime(series, errors="coerce")
                    invalid_count = (parsed.isna() & series.notna()).sum()
                    if invalid_count > 0:
                        issues[column] = f"{invalid_count} invalid datetime values"

                elif expected_type == "datetime_or_string":
                    non_null = series.notna()
                    if non_null.sum() == 0 and column not in self.OPTIONAL_FIELDS:
                        issues[column] = "All values are null"
                    else:
                        invalid = series.apply(
                            lambda x: pd.notna(x) and not (
                                isinstance(x, (datetime, pd.Timestamp)) or
                                (isinstance(x, str) and pd.notna(pd.to_datetime(x, errors="coerce")))
                            )
                        )
                        if invalid.any():
                            issues[column] = f"Found {invalid.sum()} invalid datetime or string values"

                elif expected_type in [int, float]:
                    numeric = pd.to_numeric(series, errors="coerce")
                    invalid_count = (numeric.isna() & series.notna()).sum()
                    if invalid_count > 0:
                        issues[column] = f"{invalid_count} non-numeric values"

                elif expected_type == str:
                    non_string = ~series.apply(lambda x: isinstance(x, str) or pd.isna(x))
                    if non_string.sum() > 0:
                        issues[column] = f"{non_string.sum()} non-string values"

                elif expected_type == list:
                    non_list = ~series.apply(lambda x: isinstance(x, list) or pd.isna(x))
                    if non_list.sum() > 0:
                        issues[column] = f"{non_list.sum()} non-list values"
            except Exception as e:
                logger.debug(f"Could not validate column {column}: {e}")

        return issues

    def validate_with_ge(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        store_baseline: bool = False,
        baseline_suite: Optional[Dict] = None,
    ) -> Dict:
        """Run basic statistical validation on data.

        Args:
            train_df: Training DataFrame.
            test_df: Test DataFrame.
            store_baseline: Whether to store baseline results (unused).
            baseline_suite: Baseline suite (unused, kept for compatibility).

        Returns:
            Dictionary containing validation results.
        """
        results = {
            "failed_expectations": [],
            "success_rate": 100.0,
            "total_expectations": 0,
            "failed_count": 0,
        }

        try:
            expectations_checked = 0
            expectations_passed = 0

            # Check for required columns with valid data
            required_cols = ["title", "text", "link"]
            for col in required_cols:
                if col in test_df.columns:
                    expectations_checked += 1
                    valid_count = test_df[col].notna().sum()
                    if valid_count > 0:
                        expectations_passed += 1
                    else:
                        results["failed_expectations"].append({
                            "expectation_type": "expect_column_to_exist",
                            "column": col,
                            "details": "Column has no valid values",
                        })

            # Check numeric columns for reasonable values
            for col in ["word_count", "token_count"]:
                if col in test_df.columns:
                    numeric = pd.to_numeric(test_df[col], errors="coerce")
                    expectations_checked += 1
                    valid_count = (numeric > 0).sum()
                    if valid_count > len(numeric) * 0.5:
                        expectations_passed += 1
                    else:
                        results["failed_expectations"].append({
                            "expectation_type": "expect_column_values_to_be_between",
                            "column": col,
                            "details": f"Only {valid_count}/{len(numeric)} values > 0",
                        })

            results["total_expectations"] = expectations_checked
            results["failed_count"] = expectations_checked - expectations_passed
            if expectations_checked > 0:
                results["success_rate"] = (expectations_passed / expectations_checked) * 100

            logger.info(
                f"Validation complete: {results['success_rate']:.1f}% success "
                f"({expectations_passed}/{expectations_checked})"
            )

        except Exception as e:
            logger.warning(f"Validation encountered issue: {e}")
            results.update({"error": str(e)})

        return results