"""Data validation module using basic statistical validation."""

import logging
from datetime import datetime
from typing import Dict, Optional, Union, Any

import pandas as pd

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates data types and schemas."""

    # Configuration
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

    def _extract_type_from_schema(self, schema_value: Union[type, str, Dict]) -> str:
        """Extract type string from schema value.
        
        Args:
            schema_value: Can be a type (str, int), string ("datetime"), 
                         or dict ({"type": "int", "min": 1})
        
        Returns:
            Normalized type string
        """
        # Handle dictionary format from infer_schema
        if isinstance(schema_value, dict):
            return schema_value.get("type", "unknown")
        
        # Handle direct type references
        if schema_value == str:
            return "str"
        elif schema_value == int:
            return "int"
        elif schema_value == float:
            return "float"
        elif schema_value == list:
            return "list"
        
        # Handle string type names
        return schema_value

    def validate_data_types(
        self, df: pd.DataFrame, dataset_name: str, schema: Optional[Dict] = None
    ) -> Dict:
        """Validate data types match expectations.

        Args:
            df: DataFrame to validate.
            dataset_name: Name of the dataset for logging.
            schema: Optional schema to use instead of EXPECTED_TYPES.
                   Can be in format {"col": str} or {"col": {"type": "int", "min": 1}}

        Returns:
            Dictionary mapping column names to issue descriptions.
        """
        if schema is None:
            schema = self.EXPECTED_TYPES
        
        issues = {}

        for column, schema_value in schema.items():
            if column not in df.columns:
                continue

            series = df[column]
            
            # Extract the type from schema (handles both formats)
            expected_type = self._extract_type_from_schema(schema_value)
            
            # Get min/max if available (for additional validation)
            min_val = None
            max_val = None
            if isinstance(schema_value, dict):
                min_val = schema_value.get("min")
                max_val = schema_value.get("max")

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

                elif expected_type in ["int", "float"]:
                    numeric = pd.to_numeric(series, errors="coerce")
                    invalid_count = (numeric.isna() & series.notna()).sum()
                    if invalid_count > 0:
                        issues[column] = f"{invalid_count} non-numeric values"
                    
                    # Additional range validation if min/max provided
                    if invalid_count == 0 and (min_val is not None or max_val is not None):
                        valid_numeric = numeric.dropna()
                        if len(valid_numeric) > 0:
                            if min_val is not None and (valid_numeric < min_val).any():
                                below_count = (valid_numeric < min_val).sum()
                                if column not in issues:
                                    issues[column] = f"{below_count} values below expected minimum {min_val}"
                                else:
                                    issues[column] += f"; {below_count} values below minimum"
                            
                            if max_val is not None and (valid_numeric > max_val).any():
                                above_count = (valid_numeric > max_val).sum()
                                if column not in issues:
                                    issues[column] = f"{above_count} values above expected maximum {max_val}"
                                else:
                                    issues[column] += f"; {above_count} values above maximum"

                elif expected_type == "str":
                    non_string = ~series.apply(lambda x: isinstance(x, str) or pd.isna(x))
                    if non_string.sum() > 0:
                        issues[column] = f"{non_string.sum()} non-string values"
                    
                    # Check allowed values if provided
                    if isinstance(schema_value, dict) and "allowed_values" in schema_value:
                        allowed = set(schema_value["allowed_values"])
                        non_null = series.dropna()
                        if len(non_null) > 0:
                            invalid_values = ~non_null.isin(allowed)
                            if invalid_values.any():
                                invalid_count = invalid_values.sum()
                                if column not in issues:
                                    issues[column] = f"{invalid_count} values not in allowed list"
                                else:
                                    issues[column] += f"; {invalid_count} not in allowed list"

                elif expected_type == "list":
                    non_list = ~series.apply(lambda x: isinstance(x, list) or pd.isna(x))
                    if non_list.sum() > 0:
                        issues[column] = f"{non_list.sum()} non-list values"
                        
            except Exception as e:
                logger.debug(f"Could not validate column {column}: {e}")

        return issues

    def validate_schema_completeness(
        self, df: pd.DataFrame, baseline_schema: Dict
    ) -> Dict:
        """Validate schema completeness and type matching.
        
        Args:
            df: DataFrame to validate
            baseline_schema: Baseline schema from infer_schema or baseline stats
            
        Returns:
            Dictionary with validation results
        """
        issues = {
            "missing_columns": [],
            "extra_columns": [],
            "type_mismatches": [],
            "has_issues": False
        }
        
        df_columns = set(df.columns)
        baseline_columns = set(baseline_schema.keys())
        
        # Missing columns
        missing = baseline_columns - df_columns
        if missing:
            issues["missing_columns"] = sorted(list(missing))
            issues["has_issues"] = True
        
        # Extra columns
        extra = df_columns - baseline_columns
        if extra:
            issues["extra_columns"] = sorted(list(extra))
            issues["has_issues"] = True
        
        # Type mismatches
        for col in df_columns & baseline_columns:
            expected_type = self._extract_type_from_schema(baseline_schema[col])
            actual_dtype = df[col].dtype
            
            # Check type compatibility
            type_match = self._check_type_compatibility(expected_type, actual_dtype)
            
            if not type_match:
                issues["type_mismatches"].append({
                    "column": col,
                    "expected": expected_type,
                    "actual": str(actual_dtype)
                })
                issues["has_issues"] = True
        
        return issues

    def _check_type_compatibility(self, expected_type: str, actual_dtype) -> bool:
        """Check if actual dtype is compatible with expected type.
        
        Args:
            expected_type: Expected type string
            actual_dtype: Actual pandas dtype
            
        Returns:
            True if compatible, False otherwise
        """
        if expected_type == "int" and pd.api.types.is_integer_dtype(actual_dtype):
            return True
        elif expected_type == "float" and pd.api.types.is_float_dtype(actual_dtype):
            return True
        elif expected_type == "datetime" and pd.api.types.is_datetime64_any_dtype(actual_dtype):
            return True
        elif expected_type == "datetime_or_string":
            return (pd.api.types.is_datetime64_any_dtype(actual_dtype) or 
                    pd.api.types.is_object_dtype(actual_dtype))
        elif expected_type in ["str", "list", "categorical", "object"] and pd.api.types.is_object_dtype(actual_dtype):
            return True
        
        return False

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