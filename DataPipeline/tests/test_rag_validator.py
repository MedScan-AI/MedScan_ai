"""Comprehensive tests for the DataValidator module."""

import os
import sys
from datetime import datetime
import pandas as pd
import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), '../scripts/RAG/analysis')
)
from validator import DataValidator


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame matching the actual record schema."""
    return pd.DataFrame({
        'link': [
            'https://www.cdc.gov/tb/treatment/index.html',
            'https://www.mayoclinic.org/diseases-conditions/lung-cancer/diagnosis-treatment/drc-20374627',
            'https://www.cancer.org/cancer/understanding-cancer/what-is-cancer.html',
            'https://ajronline.org/doi/full/10.2214/AJR.07.3896',
            'https://example.com/test'
        ],
        'title': [
            'Treating Tuberculosis',
            'Lung cancer - Diagnosis and treatment',
            'What Is Cancer? | Cancer Basics',
            'Just a moment...',
            'Test Title'
        ],
        'text': [
            'Treating Tuberculosis Tuberculosis (TB) CDC Skip directly...',
            'Lung cancer - Diagnosis and treatment - Mayo Clinic Cookies...',
            'What Is Cancer? Cancer Basics American Cancer Society...',
            'Just a moment...',
            'Sample test text content'
        ],
        'markdown_title': [
            'Treating Tuberculosis | Tuberculosis (TB) | CDC',
            'Lung cancer - Diagnosis and treatment - Mayo Clinic',
            'What Is Cancer? | Cancer Basics | American Cancer Society',
            None,
            'Test Title'
        ],
        'markdown_text': [
            'Treating Tuberculosis Tuberculosis (TB) CDC Skip directly...',
            'Lung cancer - Diagnosis and treatment - Mayo Clinic...',
            'What Is Cancer? Cancer Basics American Cancer Society...',
            None,
            'Sample test markdown content'
        ],
        'source_type': ['Trusted Health Portal', 'Trusted Health Portal', 'Trusted Health Portal', 'unknown', 'blog'],
        'accessed_at': [
            '2025-10-23T15:05:19.654287Z',
            '2025-10-23T15:05:28.749600Z',
            '2025-10-23T15:05:38.886091Z',
            '2025-10-23T15:05:45.123456Z',
            'invalid_datetime'
        ],
        'word_count': [2441, 55, 1230, 10, 60000],
        'token_count': [2883, 65, 1539, 12, 75000],
        'authors': [[], [], [], None, ['Author 1', 'Author 2']],
        'publish_date': [None, '2024-04-30T00:00:00', None, 'invalid', '2024-01-05'],
        'country': ['Unknown', 'Unknown', 'United States', None, 'Canada'],
        'topics': [
            ['risk factors', 'tuberculosis', 'treatment options'],
            ['lung cancer'],
            ['lung cancer', 'immunotherapy', 'treatment options', 'oncogenes', 'surgery', 'chemotherapy'],
            None,
            ['tech', 'ai']
        ]
    })


@pytest.fixture
def validator():
    """Provide a DataValidator instance."""
    return DataValidator(context_root_dir='./test_gx')


@pytest.fixture
def clean_dataframe():
    """Create a clean DataFrame with valid data."""
    return pd.DataFrame({
        'link': ['https://example.com/1', 'https://example.com/2'],
        'title': ['Title 1', 'Title 2'],
        'text': ['Content 1', 'Content 2'],
        'markdown_title': ['MD Title 1', 'MD Title 2'],
        'markdown_text': ['MD Content 1', 'MD Content 2'],
        'source_type': ['blog', 'news'],
        'accessed_at': ['2025-01-01T10:00:00Z', '2025-01-02T11:00:00Z'],
        'word_count': [100, 200],
        'token_count': [120, 250],
        'authors': [['Author1'], ['Author2']],
        'publish_date': ['2024-05-01', '2024-05-02'],
        'country': ['USA', 'Canada'],
        'topics': [['tech'], ['health']]
    })


class TestDataValidatorInitialization:
    """Tests for DataValidator initialization."""

    def test_init_default_context(self, validator):
        """Test DataValidator initialization with default context."""
        assert validator is not None
        assert hasattr(validator, 'EXPECTED_TYPES')
        assert hasattr(validator, 'validate_data_types')
        assert hasattr(validator, 'validate_with_ge')

    def test_init_custom_context(self):
        """Test DataValidator with custom context directory."""
        validator = DataValidator(context_root_dir='./custom_gx')
        assert validator.context_root_dir == './custom_gx'

    def test_expected_types_structure(self, validator):
        """Test that EXPECTED_TYPES is properly configured."""
        expected_types = validator.EXPECTED_TYPES
        
        assert 'link' in expected_types
        assert 'title' in expected_types
        assert 'text' in expected_types
        assert 'word_count' in expected_types
        assert 'token_count' in expected_types
        assert 'authors' in expected_types
        assert 'topics' in expected_types
        assert 'accessed_at' in expected_types
        assert 'publish_date' in expected_types

    def test_expected_types_values(self, validator):
        """Test that EXPECTED_TYPES have correct type values."""
        assert validator.EXPECTED_TYPES['authors'] == list
        assert validator.EXPECTED_TYPES['topics'] == list
        assert validator.EXPECTED_TYPES['text'] == str
        assert validator.EXPECTED_TYPES['link'] == str
        assert validator.EXPECTED_TYPES['accessed_at'] == 'datetime_or_string'
        assert validator.EXPECTED_TYPES['publish_date'] == 'datetime_or_string'


class TestValidateDataTypes:
    """Tests for the validate_data_types method."""

    def test_validate_clean_dataframe(self, validator, clean_dataframe):
        """Test validation of a clean DataFrame."""
        issues = validator.validate_data_types(clean_dataframe, 'clean_data')
        assert issues == {}, f"Expected no issues, but got: {issues}"

    def test_validate_empty_dataframe(self, validator):
        """Test validation of empty DataFrame."""
        df = pd.DataFrame()
        issues = validator.validate_data_types(df, 'empty')
        assert issues == {}

    def test_validate_invalid_datetime_accessed_at(self, validator):
        """Test detection of invalid datetime in accessed_at."""
        df = pd.DataFrame({
            'accessed_at': ['invalid_datetime', '2025-01-01T10:00:00Z']
        })
        issues = validator.validate_data_types(df, 'df')
        assert 'accessed_at' in issues

    def test_validate_invalid_datetime_publish_date(self, validator):
        """Test detection of invalid datetime in publish_date."""
        df = pd.DataFrame({
            'publish_date': ['invalid_date', '2024-05-01']
        })
        issues = validator.validate_data_types(df, 'df')
        assert 'publish_date' in issues

    def test_validate_invalid_numeric_word_count(self, validator):
        """Test detection of invalid word_count values."""
        df = pd.DataFrame({
            'word_count': ['notnum', 200, 'invalid']
        })
        issues = validator.validate_data_types(df, 'df')
        assert 'word_count' in issues
        assert 'non-numeric' in issues['word_count']

    def test_validate_invalid_numeric_token_count(self, validator):
        """Test detection of invalid token_count values."""
        df = pd.DataFrame({
            'token_count': ['invalid', 400, 'text']
        })
        issues = validator.validate_data_types(df, 'df')
        assert 'token_count' in issues
        assert 'non-numeric' in issues['token_count']

    def test_validate_invalid_string_link(self, validator):
        """Test detection of non-string link values."""
        df = pd.DataFrame({
            'link': [123, 'https://example.com', True]
        })
        issues = validator.validate_data_types(df, 'df')
        assert 'link' in issues
        assert 'non-string' in issues['link']

    def test_validate_invalid_string_title(self, validator):
        """Test detection of non-string title values."""
        df = pd.DataFrame({
            'title': [456, 'Title', None]
        })
        issues = validator.validate_data_types(df, 'df')
        assert 'title' in issues

    def test_validate_invalid_string_text(self, validator):
        """Test detection of non-string text values."""
        df = pd.DataFrame({
            'text': [789, 'Content', 123.45]
        })
        issues = validator.validate_data_types(df, 'df')
        assert 'text' in issues

    def test_validate_invalid_list_authors(self, validator):
        """Test detection of non-list authors values."""
        df = pd.DataFrame({
            'authors': ['not_a_list', 'string', 123]
        })
        issues = validator.validate_data_types(df, 'df')
        assert 'authors' in issues
        assert 'non-list' in issues['authors']

    def test_validate_invalid_list_topics(self, validator):
        """Test detection of non-list topics values."""
        df = pd.DataFrame({
            'topics': ['not_a_list', None, 'string']
        })
        issues = validator.validate_data_types(df, 'df')
        assert 'topics' in issues
        assert 'non-list' in issues['topics']

    def test_validate_with_schema(self, validator):
        """Test validation with custom schema."""
        df = pd.DataFrame({
            'custom_int': [1, 2, 3],
            'custom_float': [1.1, 2.2, 3.3]
        })
        schema = {
            'custom_int': int,
            'custom_float': float
        }
        issues = validator.validate_data_types(df, 'df', schema=schema)
        assert issues == {}

    def test_validate_sample_dataframe(self, validator, sample_dataframe):
        """Test validation of complex sample DataFrame."""
        issues = validator.validate_data_types(sample_dataframe, 'sample')
        
        assert 'accessed_at' in issues, "Should detect invalid datetime in accessed_at"
        assert 'publish_date' in issues, "Should detect invalid datetime in publish_date"


class TestValidateWithGE:
    """Tests for the validate_with_ge method."""

    def test_validate_with_ge_returns_dict(self, validator, clean_dataframe):
        """Test that validate_with_ge returns a dictionary."""
        result = validator.validate_with_ge(clean_dataframe, clean_dataframe)
        
        assert isinstance(result, dict)
        assert 'failed_expectations' in result
        assert 'success_rate' in result
        assert 'total_expectations' in result
        assert 'failed_count' in result

    def test_validate_with_ge_clean_data(self, validator, clean_dataframe):
        """Test validate_with_ge with clean data."""
        result = validator.validate_with_ge(clean_dataframe, clean_dataframe)
        
        assert result['success_rate'] >= 0.0
        assert result['success_rate'] <= 100.0
        assert result['total_expectations'] >= 0
        assert result['failed_count'] >= 0

    def test_validate_with_ge_returns_expectations(self, validator, clean_dataframe):
        """Test that validate_with_ge returns failed expectations."""
        result = validator.validate_with_ge(clean_dataframe, clean_dataframe)
        
        assert isinstance(result['failed_expectations'], list)
        for exp in result['failed_expectations']:
            assert isinstance(exp, dict)

    def test_validate_with_ge_handles_nan(self, validator):
        """Test validate_with_ge with NaN values."""
        df = pd.DataFrame({
            'word_count': [100, None, 150],
            'link': ['url1', None, 'url3']
        })
        result = validator.validate_with_ge(df, df)
        
        assert result is not None
        assert 'success_rate' in result

    def test_validate_with_ge_store_baseline(self, validator, clean_dataframe):
        """Test validate_with_ge with store_baseline=True."""
        result = validator.validate_with_ge(
            clean_dataframe, clean_dataframe, store_baseline=True
        )
        
        assert result is not None
        assert 'success_rate' in result

    def test_validate_with_ge_comparison(self, validator):
        """Test validate_with_ge comparing two different datasets."""
        baseline = pd.DataFrame({
            'link': ['https://example.com/1', 'https://example.com/2'],
            'title': ['Title1', 'Title2'],
            'text': ['Text1', 'Text2'],
            'word_count': [100, 200],
            'token_count': [120, 250],
            'source_type': ['blog', 'news']
        })
        
        new_data = pd.DataFrame({
            'link': ['https://example.com/3', 'https://example.com/4'],
            'title': ['Title3', 'Title4'],
            'text': ['Text3', 'Text4'],
            'word_count': [150, 250],
            'token_count': [170, 300],
            'source_type': ['blog', 'blog']
        })
        
        result = validator.validate_with_ge(baseline, new_data)
        
        assert isinstance(result, dict)
        assert 'success_rate' in result


class TestValidationIntegration:
    """Integration tests for validation pipeline."""

    def test_full_validation_pipeline(self, validator, sample_dataframe):
        """Test complete validation pipeline."""
        type_issues = validator.validate_data_types(sample_dataframe, 'sample')
        assert isinstance(type_issues, dict)
        
        ge_result = validator.validate_with_ge(sample_dataframe, sample_dataframe)
        assert isinstance(ge_result, dict)

    def test_validation_with_optional_fields(self, validator):
        """Test validation with optional fields missing."""
        df = pd.DataFrame({
            'link': ['https://example.com'],
            'title': ['Title'],
            'text': ['Content']
        })
        
        issues = validator.validate_data_types(df, 'minimal')
        assert issues == {}

    def test_validation_preserves_dataframe(self, validator, clean_dataframe):
        """Test that validation doesn't modify the DataFrame."""
        original_shape = clean_dataframe.shape
        original_values = clean_dataframe.values.copy()
        
        validator.validate_data_types(clean_dataframe, 'test')
        
        assert clean_dataframe.shape == original_shape
        assert (clean_dataframe.values == original_values).all()


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_validate_single_row_dataframe(self, validator):
        """Test validation with single row DataFrame."""
        df = pd.DataFrame({
            'link': ['https://example.com'],
            'title': ['Title'],
            'text': ['Content'],
            'word_count': [100]
        })
        
        issues = validator.validate_data_types(df, 'single_row')
        assert isinstance(issues, dict)
        
    def test_validate_with_large_numbers(self, validator):
        """Test validation with very large numbers."""
        df = pd.DataFrame({
            'word_count': [999999999, 1000000000, 5000000000],
            'token_count': [1234567890, 9876543210, 999999999999]
        })
        
        issues = validator.validate_data_types(df, 'large_numbers')
        assert issues == {}

    def test_validate_with_mixed_types(self, validator):
        """Test validation with mixed types in columns."""
        df = pd.DataFrame({
            'link': ['https://example.com', 123, None, 'another_url'],
            'word_count': [100, 'invalid', None, 200]
        })
        
        issues = validator.validate_data_types(df, 'mixed_types')
        assert 'link' in issues
        assert 'word_count' in issues


class TestDataValidationComparison:
    """Tests for comparing validation results between datasets."""

    def test_validation_result_diff(self):
        """Test manual comparison of validation results."""
        baseline = {
            'success_rate': 90.0,
            'failed_expectations': [
                {'expectation_type': 'type1', 'column': 'country'},
                {'expectation_type': 'type2', 'column': 'link'}
            ]
        }
        
        new = {
            'success_rate': 85.0,
            'failed_expectations': [
                {'expectation_type': 'type2', 'column': 'link'},
                {'expectation_type': 'type3', 'column': 'authors'}
            ]
        }
        
        success_rate_change = new['success_rate'] - baseline['success_rate']
        baseline_failures = {(f['expectation_type'], f['column']) 
                            for f in baseline['failed_expectations']}
        new_failures = {(f['expectation_type'], f['column']) 
                       for f in new['failed_expectations']}
        
        resolved_failures = baseline_failures - new_failures
        new_detected_failures = new_failures - baseline_failures
        
        assert success_rate_change == -5.0
        assert ('type3', 'authors') in new_detected_failures
        assert ('type1', 'country') in resolved_failures
        assert ('type2', 'link') in baseline_failures & new_failures


if __name__ == '__main__':
    pytest.main([__file__, '-v'])