import asyncio
import json
import sys
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# Adjust path if necessary
sys.path.insert(0, './..')

from scripts.RAG.scraper import (
    GLOBAL_METRICS,
    clean_text,
    compute_token_stats,
    detect_country,
    extract_publish_date_from_html,
    extract_text_markdown,
    extract_topic_keywords,
    fetch_url_async,
    filter_author_names,
    is_likely_person_name,
    process_single_url,
    save_record_to_file,
    strip_references,
    update_total_global_metrics,
)

@pytest.fixture
def sample_html():
    return """
    <html>
        <head>
            <title>Lung Cancer Treatment Overview</title>
            <meta name="pubdate" content="2024-01-15">
            <meta property="og:locale" content="en_US">
        </head>
        <body>
            <header>Header content</header>
            <nav>Navigation</nav>
            <h1>Understanding Lung Cancer</h1>
            <p>Lung cancer is a serious disease. <strong>NSCLC</strong> is the most common type.</p>
            <p>Treatment options include <em>chemotherapy</em> and immunotherapy.</p>
            <p>Risk factors include smoking and asbestos exposure.</p>
            <div class="cookie-banner">Accept cookies</div>
            <h2>References</h2>
            <p>[1] Smith et al, 2023</p>
            <footer>Footer content</footer>
            <script>console.log('test');</script>
        </body>
    </html>
    """

@pytest.fixture
def sample_text_with_references():
    return """
    Main content here.
    More content.

    References
    1. Author et al. (2023)
    2. Another reference
    """

@pytest.fixture
def sample_text_with_citations():
    return """
    Lung cancer affects millions [1, 2, 3].
    Studies show (Smith et al, 2023) that treatment is effective.
    Figure 1 shows the results.
    Table 2 presents the data.
    """

@pytest.fixture
def reset_global_metrics():
    GLOBAL_METRICS["total_urls"] = 0
    GLOBAL_METRICS["successfully_fetched"] = 0
    GLOBAL_METRICS["with_authors"] = 0
    GLOBAL_METRICS["with_publish_date"] = 0
    GLOBAL_METRICS["with_country"] = 0
    GLOBAL_METRICS["country_count"] = {}
    yield
    GLOBAL_METRICS["total_urls"] = 0
    GLOBAL_METRICS["successfully_fetched"] = 0
    GLOBAL_METRICS["with_authors"] = 0
    GLOBAL_METRICS["with_publish_date"] = 0
    GLOBAL_METRICS["with_country"] = 0
    GLOBAL_METRICS["country_count"] = {}

class TestNameValidation:
    def test_valid_two_word_name(self):
        assert is_likely_person_name("John Smith") == True

    def test_valid_three_word_name(self):
        assert is_likely_person_name("Mary Jane Watson") == True

    def test_invalid_url(self):
        assert is_likely_person_name("https://example.com") == False

    def test_invalid_date(self):
        assert is_likely_person_name("2024-01-15") == False

    def test_invalid_phrase_starting_with_for(self):
        assert is_likely_person_name("For Lung Cancer") == False

    def test_empty_string(self):
        assert is_likely_person_name("") == False

    def test_none_value(self):
        assert is_likely_person_name(None) == False

class TestFilterAuthors:
    def test_filter_mixed_authors(self):
        authors = ["John Smith", "For Lung Cancer", "Dr. Jane Doe", "Editorial Team"]
        filtered = filter_author_names(authors)
        assert "John Smith" in filtered
        assert "Dr. Jane Doe" in filtered
        assert "For Lung Cancer" not in filtered
        assert "Editorial Team" not in filtered

    def test_filter_empty_list(self):
        assert filter_author_names([]) == []

class TestStripReferences:
    def test_strip_references_section(self, sample_text_with_references):
        result = strip_references(sample_text_with_references)
        assert "References" not in result
        assert "Main content here" in result

class TestCleanText:
    def test_remove_bracket_citations(self, sample_text_with_citations):
        result = clean_text(sample_text_with_citations)
        assert "[1, 2, 3]" not in result

class TestComputeTokenStats:
    def test_word_count(self):
        text = "This is a simple test"
        words, tokens = compute_token_stats(text)
        assert words == 5

class TestDetectCountry:
    def test_detect_from_domain_map(self):
        url = "https://www.nih.gov/health/article"
        result = detect_country(url)
        assert result == "United States"

class TestExtractPublishDate:
    def test_extract_from_meta_tag(self, sample_html):
        result = extract_publish_date_from_html(sample_html)
        assert result == "2024-01-15"

class TestExtractTopicKeywords:
    def test_extract_lung_cancer_keywords(self):
        text = "Patient diagnosed with lung cancer and NSCLC"
        result = extract_topic_keywords(text)
        assert "lung cancer" in result
        assert "nsclc" in result

class TestExtractTextMarkdown:
    def test_removes_script_tags(self, sample_html):
        title, markdown = extract_text_markdown(sample_html)
        assert "console.log" not in markdown

class TestSaveRecordToFile:
    def test_save_valid_record(self, tmp_path):
        filename = tmp_path / "test_output.jsonl"
        record = {"link": "https://example.com", "title": "Test", "text": "Content"}
        save_record_to_file(record, str(filename))
        assert filename.exists()

class TestUpdateGlobalMetrics:
    def test_update_with_error(self, reset_global_metrics):
        record = {"link": "https://example.com", "error": "Failed"}
        update_total_global_metrics(record)
        assert GLOBAL_METRICS["total_urls"] == 1
        assert GLOBAL_METRICS["successfully_fetched"] == 0

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])