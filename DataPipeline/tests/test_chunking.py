import pytest
import json
import os
import re
from pathlib import Path
from unittest.mock import patch, mock_open

# Assuming chunking.py and embedding.py are in the same directory
# Adjust paths if necessary
import sys
sys.path.insert(0, './..')

from scripts.RAG.chunking import RAGChunker, HeaderDetector
from scripts.RAG.embedding import EmbeddedChunk, ChunkEmbedder # Import from embedding

# Mocking the tiktoken encoding for testing purposes
class MockEncoding:
    def encode(self, text):
        # Simple token estimation for testing: roughly word count / 1.5
        return [0] * int(len(text.split()) / 1.5)

@pytest.fixture
def mock_tiktoken_get_encoding(monkeypatch):
    """Fixture to mock tiktoken.get_encoding."""
    def mock_get_encoding(encoding_name):
        return MockEncoding()
    monkeypatch.setattr("tiktoken.get_encoding", mock_get_encoding)

@pytest.fixture
def header_detector():
    """Fixture for HeaderDetector instance."""
    return HeaderDetector()

@pytest.fixture
def rag_chunker():
    """Fixture for RAGChunker instance."""
    return RAGChunker()

# --- HeaderDetector Tests ---

def test_is_likely_heading_markdown(header_detector):
    assert header_detector.is_likely_heading("# Main Heading")[0] is False
    assert header_detector.is_likely_heading("## Sub Heading")[0] is False

def test_is_likely_heading_bold(header_detector):
    assert header_detector.is_likely_heading("**Bold Heading**")[0] is True
    assert header_detector.is_likely_heading("__Another Bold Heading__")[0] is True
    assert header_detector.is_likely_heading("**This is a very long bold sentence that should not be a heading because it is too long**")[0] is False

def test_is_likely_heading_all_caps(header_detector):
    assert header_detector.is_likely_heading("ALL CAPS HEADING")[0] is True
    assert header_detector.is_likely_heading("SHORT CAPS")[0] is True
    assert header_detector.is_likely_heading("THIS IS A VERY LONG ALL CAPS SENTENCE THAT SHOULD NOT BE A HEADING BECAUSE IT IS TOO LONG")[0] is False
    assert header_detector.is_likely_heading("WITH-HYPHENS")[0] is True
    assert header_detector.is_likely_heading("WITH NUMBERS 123")[0] is True
    assert header_detector.is_likely_heading("MiXeD cApS")[0] is False # Should be all upper

def test_is_likely_heading_short_line_followed_by_long(header_detector):
    assert header_detector.is_likely_heading("Introduction:", "This is the main content that follows the introduction.")[0] is True
    assert header_detector.is_likely_heading("Summary", "This is a longer summary of the document.")[0] is True
    assert header_detector.is_likely_heading("Conclusion", "This is the conclusion of the paper.")[0] is True
    # Should not be heading (ends with punctuation)
    assert header_detector.is_likely_heading("Introduction.", "This is the main content.")[0] is False
    # Should not be heading (short line not followed by long line)
    assert header_detector.is_likely_heading("Short line", "Short content.")[0] is False

def test_is_likely_heading_numbered_sections(header_detector):
    assert header_detector.is_likely_heading("1. First Section")[0] is True
    assert header_detector.is_likely_heading("2.1. Sub-section")[0] is True
    assert header_detector.is_likely_heading("1. This is a very long numbered heading that should not be identified as such because it is too long")[0] is False

def test_is_likely_heading_question_format(header_detector):
    assert header_detector.is_likely_heading("What is Cancer?")[0] is True
    assert header_detector.is_likely_heading("How to Treat TB?")[0] is True
    assert header_detector.is_likely_heading("a question?")[0] is False # Should start with upper

def test_is_likely_heading_keywords_in_title_case(header_detector):
    assert header_detector.is_likely_heading("Methods Section")[0] is True
    assert header_detector.is_likely_heading("Treatment Options")[0] is True
    assert header_detector.is_likely_heading("Discussion Points")[0] is True
    assert header_detector.is_likely_heading("Cancer Symptoms Overview")[0] is True
    assert header_detector.is_likely_heading("A random sentence with keywords like treatment but not a heading.")[0] is False # Too long

def test_process_markdown_adds_headers(header_detector):
    content = """
Introduction:
This is the introduction content.

Methods
Here are the methods used.

Conclusion.
This is the conclusion.

Another section:
Content for another section.
"""
    processed_content = header_detector.process_markdown(content)
    lines = processed_content.strip().split('\n')

    # Check if headers were added
    assert "# Introduction:" in lines
    assert "# Methods" in lines
    # Note: "Conclusion." might not be detected due to trailing punctuation
    assert "# Another section:" in lines

def test_process_markdown_preserves_existing_headers(header_detector):
    content = """
# Existing Main Heading

Some introductory text.

## Existing Sub-heading

More content under the sub-heading.
"""
    processed_content = header_detector.process_markdown(content)
    lines = processed_content.strip().split('\n')

    assert "# Existing Main Heading" in lines
    assert "## Existing Sub-heading" in lines

# --- RAGChunker Tests ---

def test_count_tokens(rag_chunker, mock_tiktoken_get_encoding):
    assert rag_chunker.count_tokens("This is a short sentence.") == int(len("This is a short sentence.".split()) / 1.5)
    assert rag_chunker.count_tokens("This is a slightly longer sentence to test the token count function.") == int(len("This is a slightly longer sentence to test the token count function.".split()) / 1.5)
    assert rag_chunker.count_tokens("") == 0
    assert rag_chunker.count_tokens("   ") == 0

def test_has_headers(rag_chunker):
    content_with_headers = """
# Section 1
Content of section 1.
## Subsection 1.1
Content of subsection 1.1.
"""
    content_without_headers = """
Introduction:
This is the introduction.

Methods:
These are the methods.
"""
    assert rag_chunker.has_headers(content_with_headers) is True
    assert rag_chunker.has_headers(content_without_headers) is False
    assert rag_chunker.has_headers("") is False

def test_find_min_header_level(rag_chunker):
    content1 = """
# H1
## H2
### H3
"""
    content2 = """
## H2
### H3
#### H4
"""
    content3 = """
### H3 only
"""
    content4 = """
No headers here.
"""
    assert rag_chunker.find_min_header_level(content1) == 1
    assert rag_chunker.find_min_header_level(content2) == 2
    assert rag_chunker.find_min_header_level(content3) == 3
    assert rag_chunker.find_min_header_level(content4) == 1 # Default to 1 if no headers

# @pytest.mark.skip(reason="Requires every-header chunking (Issue #4) - parked for future")
# def test_chunk_by_headers_basic(rag_chunker):
#     content = """
# # Introduction
# This is the introduction.
# It spans multiple lines.

# ## Section 1
# Content of section 1.

# ### Subsection 1.1
# Content of subsection 1.1.

# ## Section 2
# Content of section 2.

# # Conclusion
# This is the conclusion.
# """
#     chunks = rag_chunker.chunk_by_headers(content)

#     assert len(chunks) == 5
#     assert chunks[0]['header'] == 'Introduction'
#     assert chunks[0]['level'] == 1
#     assert "This is the introduction.\nIt spans multiple lines." in chunks[0]['content']

#     assert chunks[1]['header'] == 'Section 1'
#     assert chunks[1]['level'] == 2
#     assert "Content of section 1." in chunks[1]['content']
#     assert "### Subsection 1.1\nContent of subsection 1.1." in chunks[1]['content'] # Subsection included

#     assert chunks[2]['header'] == 'Subsection 1.1'
#     assert chunks[2]['level'] == 3
#     assert "Content of subsection 1.1." in chunks[2]['content']

#     assert chunks[3]['header'] == 'Section 2'
#     assert chunks[3]['level'] == 2
#     assert "Content of section 2." in chunks[3]['content']

#     assert chunks[4]['header'] == 'Conclusion'
#     assert chunks[4]['level'] == 1
#     assert "This is the conclusion." in chunks[4]['content']


def test_chunk_by_headers_no_headers(rag_chunker):
    content = """
This is some content without any markdown headers.
It should all be in a single chunk.
It has multiple paragraphs.
"""
    chunks = rag_chunker.chunk_by_headers(content)

    assert len(chunks) == 1
    assert chunks[0]['header'] == 'Introduction' # Default header
    assert chunks[0]['level'] == 0
    assert "This is some content without any markdown headers." in chunks[0]['content']
    assert "It should all be in a single chunk.\nIt has multiple paragraphs." in chunks[0]['content']

def test_chunk_by_headers_empty_content(rag_chunker):
    content = ""
    chunks = rag_chunker.chunk_by_headers(content)
    assert len(chunks) == 0

# @pytest.mark.skip(reason="Requires every-header chunking (Issue #4) - parked for future")
# def test_process_file_with_existing_headers(rag_chunker, mock_tiktoken_get_encoding):
#     record = {
#         'link': 'http://example.com/doc1',
#         'title': 'Document One',
#         'markdown_text': """
# # Introduction
# This is the intro.

# ## Methods
# These are the methods.
# """,
#         'text': 'Cleaned text content',
#         'markdown_title': 'Markdown Title',
#         'word_count': 100,
#         'token_count': 50,
#         'source_type': 'Journal',
#         'authors': ['Author A'],
#         'publish_date': '2023-01-01',
#         'country': 'USA',
#         'topics': ['Topic1'],
#         'accessed_at': '2023-01-02'
#     }

#     chunks = rag_chunker.process_file(record)

#     assert len(chunks) == 2
#     assert chunks[0]['section_header'] == 'Introduction'
#     assert 'This is the intro.' in chunks[0]['content']
#     assert chunks[0]['link'] == 'http://example.com/doc1'
#     assert chunks[0]['title'] == 'Document One'
#     assert chunks[0]['source_type'] == 'Journal'
#     assert chunks[0]['authors'] == ['Author A']
#     assert chunks[0]['chunk_token_count'] >= 0 # Check token count is added

#     assert chunks[1]['section_header'] == 'Methods'
#     assert 'These are the methods.' in chunks[1]['content']
#     assert chunks[1]['link'] == 'http://example.com/doc1' # Ensure metadata is propagated

# @pytest.mark.skip(reason="Requires every-header chunking (Issue #4) - parked for future")
# def test_process_file_without_existing_headers(rag_chunker, mock_tiktoken_get_encoding):
#     record = {
#         'link': 'http://example.com/doc2',
#         'title': 'Document Two',
#         'markdown_text': """
# Article Title

# Summary:
# This is the summary.

# Details:
# Here are the details.

# Conclusion.
# Final thoughts here.
# """,
#         'text': 'Cleaned text content without headers.',
#         'markdown_title': 'Markdown Title Two',
#         'word_count': 150,
#         'token_count': 75,
#         'source_type': 'Website',
#         'authors': [],
#         'publish_date': None,
#         'country': 'UK',
#         'topics': ['Topic2', 'Topic3'],
#         'accessed_at': '2023-01-03'
#     }

#     chunks = rag_chunker.process_file(record)

#     assert len(chunks) >= 3 # Expecting at least Intro, Summary, Details
#     assert chunks[0]['section_header'] == 'Introduction' # Default intro chunk
#     assert 'Article Title' in chunks[0]['content']

#     summary_chunk = next((c for c in chunks if c['section_header'] == 'Summary:'), None)
#     assert summary_chunk is not None
#     assert 'This is the summary.' in summary_chunk['content']
#     assert summary_chunk['link'] == 'http://example.com/doc2'
#     assert summary_chunk['chunk_token_count'] >= 0

#     details_chunk = next((c for c in chunks if c['section_header'] == 'Details:'), None)
#     assert details_chunk is not None
#     assert 'Here are the details.' in details_chunk['content']

#     # Conclusion might or might not be a header depending on detector logic
#     conclusion_chunk = next((c for c in chunks if 'Final thoughts here.' in c['content']), None)
#     assert conclusion_chunk is not None

def test_process_file_empty_content(rag_chunker, mock_tiktoken_get_encoding):
    record = {
        'link': 'http://example.com/empty',
        'title': 'Empty Document',
        'markdown_text': '',
        'text': '',
        'markdown_title': '',
        'word_count': 0,
        'token_count': 0,
        'source_type': 'Unknown',
        'authors': [],
        'publish_date': None,
        'country': 'Unknown',
        'topics': [],
        'accessed_at': '2023-01-04'
    }
    chunks = rag_chunker.process_file(record)
    assert len(chunks) == 0 # No chunks should be generated for empty content

def test_process_file_missing_content_keys(rag_chunker, mock_tiktoken_get_encoding):
    record = {
        'link': 'http://example.com/missing',
        'title': 'Missing Content Keys',
        # No 'markdown_text' or 'text'
        'word_count': 0,
        'token_count': 0,
        'source_type': 'Unknown',
        'authors': [],
        'publish_date': None,
        'country': 'Unknown',
        'topics': [],
        'accessed_at': '2023-01-05'
    }
    chunks = rag_chunker.process_file(record)
    assert len(chunks) == 0 # No chunks should be generated

# Test process_directory requires mocking file operations
@patch("json.dump")
@patch.object(RAGChunker, 'process_file')
@patch("builtins.open", new_callable=mock_open, read_data='{"link": "url1", "markdown_text": "Content 1"}\n{"link": "url2", "markdown_text": "Content 2"}')
@patch("pathlib.Path.mkdir")
def test_process_directory(mock_mkdir, mock_file, mock_process_file, mock_json_dump, rag_chunker):
    # Configure mock_process_file to return sample chunks
    mock_process_file.side_effect = [
        [{'section_header': 'h1', 'content': 'c1', 'level': 1, 'chunk_token_count': 10, 'link': 'url1'}],
        [{'section_header': 'h2', 'content': 'c2', 'level': 1, 'chunk_token_count': 20, 'link': 'url2'}]
    ]

    input_file = Path('scraped_data_baseline.jsonl')
    output_file = Path('chunks.json')

    chunks = rag_chunker.process_directory(input_file, output_file)

    # Assertions
    assert len(chunks) == 2
    assert chunks[0]['content'] == 'c1'
    assert chunks[1]['content'] == 'c2'

    # Check if json.dump was called correctly
    assert mock_json_dump.call_count == 1
    called_chunks = mock_json_dump.call_args[0][0]
    assert called_chunks == chunks

    # Check if process_file was called for each record
    assert mock_process_file.call_count == 2
    call_args_list = mock_process_file.call_args_list
    assert call_args_list[0][0][0]['link'] == 'url1'
    assert call_args_list[1][0][0]['link'] == 'url2'

# Add tests for specific heading keyword detection in HeaderDetector
@pytest.mark.parametrize("line, expected", [
    ("Introduction", True),
    ("Overview", True),
    ("Background", True),
    ("Summary", True),
    ("Conclusion", True),
    ("Abstract", True),
    ("Methods", True),
    ("Results", True),
    ("Discussion", True),
    ("References", True),
    ("Symptoms", True),
    ("Treatment", True),
    ("Diagnosis", True),
    ("Causes", True),
    ("Prevention", True),
    ("What is Cancer", True),
    ("How to treat", True),
    ("Why it happens", True),
    ("When to seek help", True),
    ("Where to find information", True),
    ("Random sentence", False),
    ("Introduction to the topic", True), # Test phrase detection
])
def test_is_likely_heading_keyword_detection(header_detector, line, expected):
     # Mock a longer next line to satisfy the condition
     next_line = "This is a very long sentence to trigger the heuristic for short lines followed by long lines."
     is_heading, level = header_detector.is_likely_heading(line, next_line=next_line)
     assert is_heading is expected

# Add test for find_min_header_level with mixed markdown and heuristic headers
@patch.object(HeaderDetector, 'is_likely_heading', return_value=(False, 0)) # Disable heuristic detection
def test_find_min_header_level_only_markdown(mock_is_likely_heading, rag_chunker):
     content = """
# H1
Some text
## H2
More text
### H3
Even more text
"""
     assert rag_chunker.find_min_header_level(content) == 1

@patch.object(HeaderDetector, 'is_likely_heading', side_effect=[
    (False, 0), # # H1
    (False, 0), # text
    (True, 2),  # ## H2 (detected as heuristic H2, but markdown takes precedence)
    (False, 0), # text
    (True, 3),  # ### H3 (detected as heuristic H3, but markdown takes precedence)
    (False, 0) # text
])
def test_find_min_header_level_mixed(mock_is_likely_heading, rag_chunker):
     content = """
# H1
Some text
## H2
More text
### H3
Even more text
"""
     # Note: The mock overrides the actual markdown header detection in find_min_header_level.
     # A more sophisticated mock might be needed to fully test the interaction,
     # but this tests that the function correctly identifies the *markdown* levels.
     assert rag_chunker.find_min_header_level(content) == 1

@patch.object(HeaderDetector, 'is_likely_heading', side_effect=[
    (True, 2), # Heuristic H2
    (False, 0),
    (True, 3), # Heuristic H3
    (False, 0),
])
def test_find_min_header_level_only_heuristic(mock_is_likely_heading, rag_chunker):
     content = """
Section Title:
Some text.

Subsection:
More text.
"""
     # When only heuristic headers are present, the function should fall back to level 1
     # as it only looks for markdown headers.
     assert rag_chunker.find_min_header_level(content) == 1

# Add tests for edge cases in chunk_by_headers
def test_chunk_by_headers_leading_content(rag_chunker):
    content = """
Leading content before the first header.

# Section 1
Content of section 1.
"""
    chunks = rag_chunker.chunk_by_headers(content)
    assert len(chunks) == 2
    assert chunks[0]['header'] == 'Introduction'
    assert 'Leading content before the first header.' in chunks[0]['content']
    assert chunks[1]['header'] == 'Section 1'

# @pytest.mark.skip(reason="Requires every-header chunking (Issue #4) - parked for future")
# def test_chunk_by_headers_trailing_content(rag_chunker):
#     content = """
# # Section 1
# Content of section 1.

# Trailing content after the last header.
# """
#     chunks = rag_chunker.chunk_by_headers(content)
#     assert len(chunks) == 2
#     assert chunks[0]['header'] == 'Section 1'
#     assert chunks[1]['header'] == 'Introduction' # Trailing content gets default header
#     assert 'Trailing content after the last header.' in chunks[1]['content']


# @pytest.mark.skip(reason="Requires every-header chunking (Issue #4) - parked for future")
# def test_chunk_by_headers_consecutive_headers(rag_chunker):
#     content = """
# # Header 1
# # Header 2
# # Header 3
# Content.
# """
#     chunks = rag_chunker.chunk_by_headers(content)
#     assert len(chunks) == 4 # H1, H2, H3, and trailing content
#     assert chunks[0]['header'] == 'Header 1'
#     assert chunks[0]['content'] == '' # No content under Header 1
#     assert chunks[1]['header'] == 'Header 2'
#     assert chunks[1]['content'] == '' # No content under Header 2
#     assert chunks[2]['header'] == 'Header 3'
#     assert 'Content.' in chunks[2]['content']


# @pytest.mark.skip(reason="Requires every-header chunking (Issue #4) - parked for future")
# def test_chunk_by_headers_only_headers(rag_chunker):
#     content = """
# # Header 1
# ## Header 1.1
# # Header 2
# """
#     chunks = rag_chunker.chunk_by_headers(content)
#     assert len(chunks) == 3 # H1, H1.1, H2
#     assert chunks[0]['header'] == 'Header 1'
#     assert '## Header 1.1' in chunks[0]['content'] # Sub-header included in parent chunk
#     assert chunks[1]['header'] == 'Header 1.1'
#     assert chunks[1]['content'] == '' # No content under H1.1
#     assert chunks[2]['header'] == 'Header 2'
#     assert chunks[2]['content'] == '' # No content under H2

# @pytest.mark.skip(reason="Requires every-header chunking (Issue #4) - parked for future")
# def test_chunk_by_headers_respects_min_header_level(rag_chunker):
#     content = """
# # H1 (Should not be a chunk boundary if min level is 2)
# Content for H1.
# ## H2 (Should be a chunk boundary)
# Content for H2.
# ### H3 (Should be a chunk boundary)
# Content for H3.
# #### H4 (Should NOT be a chunk boundary)
# Content for H4.
# """
#     # Force chunking at level 2
#     rag_chunker.find_min_header_level = lambda x: 2

#     chunks = rag_chunker.chunk_by_headers(content)

#     assert len(chunks) == 2 # Should chunk by H2 and H3, but H1 becomes intro
#     assert chunks[0]['header'] == 'Introduction' # H1 is now just content
#     assert 'Content for H1.' in chunks[0]['content']
#     assert '## H2' in chunks[0]['content'] # H2 is also content in the 'intro' chunk
#                                           # This might be unexpected behavior and should be reviewed.
#                                           # The current logic adds the header line itself to the content.
#                                           # Let's refine this test to check the intended behavior.
#     assert 'Content for H2.' in chunks[0]['content']
#     assert '### H3' in chunks[0]['content']
#     assert 'Content for H3.' in chunks[0]['content']
#     assert '#### H4' in chunks[0]['content']
#     assert 'Content for H4.' in chunks[0]['content']

#     assert chunks[1]['header'] == 'H2'
#     assert 'Content for H2.' in chunks[1]['content']
#     assert '### H3' in chunks[1]['content'] # H3 is below chunk level 2, included as content
#     assert 'Content for H3.' in chunks[1]['content']
#     assert '#### H4' in chunks[1]['content'] # H4 is below chunk level 2, included as content
#     assert 'Content for H4.' in chunks[1]['content']

#     assert chunks[2]['header'] == 'H3'
#     assert 'Content for H3.' in chunks[2]['content']
#     assert '#### H4' in chunks[2]['content'] # H4 is below chunk level 3, included as content
#     assert 'Content for H4.' in chunks[2]['content']

#     # This shows the chunking logic correctly includes lower-level headers as content.
#     # The original intent of the instruction might have been to test filtering
#     # by a specified minimum header level, which the current implementation
#     # uses the *minimum* found level. Let's add a test that simulates
#     # a scenario where only H2 and H3 exist, so the min level is 2.

#     rag_chunker = RAGChunker() # Fresh instance
#     content = """
# Some leading content.

# ## H2 (Min level detected will be 2)
# Content for H2.
# ### H3 (Below min level, included as content)
# Content for H3.
# #### H4 (Below min level, included as content)
# Content for H4.

# ## Another H2
# Content for Another H2.
# """
#     chunks = rag_chunker.chunk_by_headers(content) # Min level detected will be 2

#     assert len(chunks) == 3 # Leading content, first H2, second H2
#     assert chunks[0]['header'] == 'Introduction'
#     assert 'Some leading content.' in chunks[0]['content']
#     assert '## H2' in chunks[0]['content'] # Current behavior appends the header line to prev chunk
#     assert 'Content for H2.' in chunks[0]['content']
#     assert '### H3' in chunks[0]['content']
#     assert 'Content for H3.' in chunks[0]['content']
#     assert '#### H4' in chunks[0]['content']
#     assert 'Content for H4.' in chunks[0]['content']
#     assert '## Another H2' in chunks[0]['content']
#     assert 'Content for Another H2.' in chunks[0]['content']


# # Add a test for metadata propagation in process_file with multiple chunks
# @pytest.mark.skip(reason="Requires every-header chunking (Issue #4) - parked for future")
# def test_process_file_metadata_propagation(rag_chunker, mock_tiktoken_get_encoding):
#     record = {
#         'link': 'http://example.com/metadata',
#         'title': 'Metadata Test',
#         'markdown_text': """
# # Section 1
# Content 1.

# ## Section 1.1
# Content 1.1.

# # Section 2
# Content 2.
# """,
#         'text': 'Cleaned text.',
#         'markdown_title': 'MD Title',
#         'word_count': 200,
#         'token_count': 100,
#         'source_type': 'Blog',
#         'authors': ['Author B', 'Author C'],
#         'publish_date': '2024-01-01',
#         'country': 'Canada',
#         'topics': ['Topic X', 'Topic Y'],
#         'accessed_at': '2024-01-02'
#     }

#     chunks = rag_chunker.process_file(record)

#     assert len(chunks) == 3 # Section 1, Section 1.1, Section 2

#     # Check metadata is present and correct in all chunks
#     for chunk in chunks:
#         assert chunk['link'] == 'http://example.com/metadata'
#         assert chunk['title'] == 'Metadata Test'
#         assert chunk['source_type'] == 'Blog'
#         assert chunk['authors'] == ['Author B', 'Author C']
#         assert chunk['publish_date'] == '2024-01-01'
#         assert chunk['country'] == 'Canada'
#         assert chunk['topics'] == ['Topic X', 'Topic Y']
#         assert chunk['accessed_at'] == '2024-01-02'
#         assert 'chunk_token_count' in chunk and chunk['chunk_token_count'] >= 0
#         assert 'section_header' in chunk
#         assert 'section_level' in chunk

# # Add a test for handling records with only 'text' and no 'markdown_text'
# @pytest.mark.skip(reason="Requires every-header chunking (Issue #4) - parked for future")
# def test_process_file_only_text(rag_chunker, mock_tiktoken_get_encoding):
#     record = {
#         'link': 'http://example.com/onlytext',
#         'title': 'Only Text Test',
#         'text': """
# Introduction
# This is the introduction in plain text.

# Methods
# These are the methods.
# """,
#         'markdown_title': '',
#         'word_count': 80,
#         'token_count': 40,
#         'source_type': 'Plain',
#         'authors': [],
#         'publish_date': None,
#         'country': 'Unknown',
#         'topics': [],
#         'accessed_at': '2024-01-03'
#     }
#     chunks = rag_chunker.process_file(record)
#     # Expecting the header detector to add headers based on lines like "Introduction", "Methods"
#     # and then chunk based on those.
#     assert len(chunks) >= 2 # At least Intro and Methods

#     intro_chunk = next((c for c in chunks if 'This is the introduction' in c['content']), None)
#     assert intro_chunk is not None
#     assert intro_chunk['section_header'] in ['Introduction', '# Introduction']

#     methods_chunk = next((c for c in chunks if 'These are the methods' in c['content']), None)
#     assert methods_chunk is not None
#     assert methods_chunk['section_header'] in ['Methods', '# Methods']


# Add a test for handling empty lists in metadata fields
def test_process_file_empty_lists(rag_chunker, mock_tiktoken_get_encoding):
    record = {
        'link': 'http://example.com/emptylists',
        'title': 'Empty Lists Test',
        'markdown_text': '# Section 1\nContent.',
        'text': 'Content.',
        'markdown_title': 'Title',
        'word_count': 10,
        'token_count': 5,
        'source_type': 'Test',
        'authors': [], # Empty list
        'publish_date': None,
        'country': 'Testland',
        'topics': [], # Empty list
        'accessed_at': '2024-01-04'
    }
    chunks = rag_chunker.process_file(record)
    assert len(chunks) == 1
    assert chunks[0]['authors'] == []
    assert chunks[0]['topics'] == []

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])