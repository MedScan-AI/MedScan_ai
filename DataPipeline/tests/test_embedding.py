import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest

# Adjust path if necessary
sys.path.insert(0, './..')

from scripts.RAG.embedding import ChunkEmbedder, EmbeddedChunk

# Mock the SentenceTransformer class
class MockSentenceTransformer:
    def __init__(self, model_name):
        self.model_name = model_name
        self._embedding_dimension = 768 # Default dimension for mock

    def encode(self, sentences, convert_to_numpy=True):
        if isinstance(sentences, str):
            sentences = [sentences]

        # Check for the specific empty combined text ". . "
        if len(sentences) == 1 and sentences[0].strip() == '. .':
             if convert_to_numpy:
                 return np.array([]) # Return empty numpy array for this specific empty case
             return [] # Return empty list

        # Handle truly empty input if the combined text is just whitespace or empty
        if not any(s.strip() for s in sentences):
             if convert_to_numpy:
                 return np.array([]) # Return empty numpy array for no text
             return [] # Return empty list


        # Return dummy embeddings based on the number of sentences
        embeddings = np.random.rand(len(sentences), self._embedding_dimension)

        # If only one sentence was passed and numpy output requested, return 1D array
        if len(sentences) == 1 and convert_to_numpy:
             return embeddings[0]

        if convert_to_numpy:
            return embeddings
        return embeddings.tolist()

    def get_sentence_embedding_dimension(self):
        return self._embedding_dimension

@pytest.fixture
def mock_sentence_transformer(monkeypatch):
    """Fixture to mock the SentenceTransformer class."""
    monkeypatch.setattr("scripts.RAG.embedding.SentenceTransformer", MockSentenceTransformer)

@pytest.fixture
def chunk_embedder(mock_sentence_transformer):
    """Fixture for ChunkEmbedder instance with mocked model."""
    return ChunkEmbedder(model_name='mock-model')

@pytest.fixture
def sample_chunk_data():
    """Fixture for sample chunk data dictionary."""
    return {
        'link': 'http://example.com/doc1',
        'title': 'Document One',
        'content': 'This is the content of document one.',
        'source_type': 'Journal',
        'authors': ['Author A'],
        'publish_date': '2023-01-01',
        'country': 'USA',
        'topics': ['Topic1'],
        'accessed_at': '2023-01-02',
        'chunk_token_count': 50,
        'section_header': 'Introduction',
        'section_level': 1
    }

@pytest.fixture
def sample_embedded_chunk(sample_chunk_data):
     """Fixture for a sample EmbeddedChunk object."""
     embedding = np.random.rand(768)
     return EmbeddedChunk(
         chunk_id='chunk_0',
         title=sample_chunk_data['title'],
         content=sample_chunk_data['content'],
         embedding=embedding,
         metadata={k: v for k, v in sample_chunk_data.items() if k not in ['title', 'content']} # Include chunk_token_count here
     )

# --- ChunkEmbedder Tests ---

def test_chunk_embedder_initialization(chunk_embedder):
    assert chunk_embedder.model_name == 'mock-model'
    assert chunk_embedder.model is None

def test_load_model_success(chunk_embedder):
    chunk_embedder.load_model()
    assert isinstance(chunk_embedder.model, MockSentenceTransformer)
    assert chunk_embedder.model.model_name == 'mock-model'
    assert chunk_embedder.model.get_sentence_embedding_dimension() == 768

@patch("scripts.RAG.embedding.SentenceTransformer")
def test_load_model_failure(mock_st_constructor):
    mock_st_constructor.side_effect = Exception("Failed to load model")
    embedder = ChunkEmbedder(model_name='invalid-model')
    with pytest.raises(Exception, match="Failed to load model"):
        embedder.load_model()
    assert embedder.model is None

def test_create_combined_text_basic(chunk_embedder, sample_chunk_data):
    combined = chunk_embedder.create_combined_text(sample_chunk_data)
    expected = "Document One. Document One. This is the content of document one."
    assert combined == expected

def test_create_combined_text_empty_title(chunk_embedder, sample_chunk_data):
    chunk_data = sample_chunk_data.copy()
    chunk_data['title'] = ''
    combined = chunk_embedder.create_combined_text(chunk_data)
    expected = ". . This is the content of document one."
    assert combined == expected

def test_create_combined_text_empty_content(chunk_embedder, sample_chunk_data):
    chunk_data = sample_chunk_data.copy()
    chunk_data['content'] = ''
    combined = chunk_embedder.create_combined_text(chunk_data)
    expected = "Document One. Document One. "
    assert combined == expected

def test_create_combined_text_empty_title_and_content(chunk_embedder, sample_chunk_data):
    chunk_data = sample_chunk_data.copy()
    chunk_data['title'] = ''
    chunk_data['content'] = ''
    combined = chunk_embedder.create_combined_text(chunk_data)
    expected = ". . " # This is the current behavior, which the mock needs to handle
    assert combined == expected

def test_embed_chunk_success(chunk_embedder, sample_chunk_data):
    chunk_embedder.load_model() # Ensure model is loaded
    embedded = chunk_embedder.embed_chunk(sample_chunk_data, 'chunk_abc')

    assert isinstance(embedded, EmbeddedChunk)
    assert embedded.chunk_id == 'chunk_abc'
    assert embedded.title == sample_chunk_data['title']
    assert embedded.content == sample_chunk_data['content']
    assert isinstance(embedded.embedding, np.ndarray)
    assert embedded.embedding.shape == (768,) # Check embedding dimension
    # Ensure metadata is correctly extracted, including 'chunk_token_count', 'section_header', 'section_level'
    expected_metadata_keys = {k for k in sample_chunk_data.keys() if k not in ['title', 'content']}
    assert set(embedded.metadata.keys()) == expected_metadata_keys
    for key in expected_metadata_keys:
        assert embedded.metadata[key] == sample_chunk_data[key]


def test_embed_chunk_empty_text(chunk_embedder, sample_chunk_data):
    chunk_embedder.load_model() # Ensure model is loaded
    chunk_data = sample_chunk_data.copy()
    chunk_data['content'] = ''
    chunk_data['title'] = '' # Combined text will be ". . "

    embedded = chunk_embedder.embed_chunk(chunk_data, 'chunk_empty')
    # The mock is adjusted to return empty array for ". . " after stripping
    assert embedded is None # Should return None if embedding fails or is empty

def test_embed_chunks_success(chunk_embedder, sample_chunk_data):
    chunk_embedder.load_model() # Ensure model is loaded
    chunks_list = [sample_chunk_data.copy() for _ in range(5)] # List of 5 sample chunks
    embedded_list = chunk_embedder.embed_chunks(chunks_list)

    assert isinstance(embedded_list, list)
    assert len(embedded_list) == 5 # Expect 5 embedded chunks now
    for i, embedded in enumerate(embedded_list):
        assert isinstance(embedded, EmbeddedChunk)
        assert embedded.chunk_id == f'chunk_{i}'
        assert embedded.title == sample_chunk_data['title']
        assert embedded.content == sample_chunk_data['content']
        assert isinstance(embedded.embedding, np.ndarray)
        assert embedded.embedding.shape == (768,)
        # Ensure metadata is correctly extracted
        expected_metadata_keys = {k for k in sample_chunk_data.keys() if k not in ['title', 'content']}
        assert set(embedded.metadata.keys()) == expected_metadata_keys
        for key in expected_metadata_keys:
            assert embedded.metadata[key] == sample_chunk_data[key]


def test_embed_chunks_with_empty_and_valid(chunk_embedder, sample_chunk_data):
    chunk_embedder.load_model() # Ensure model is loaded
    chunks_list = [
        sample_chunk_data.copy(),
        {**sample_chunk_data.copy(), 'title': '', 'content': ''}, # Empty chunk (combined text is ". . ")
        sample_chunk_data.copy(),
        {**sample_chunk_data.copy(), 'content': 'Only content'}, # Chunk with only content
        sample_chunk_data.copy()
    ]
    embedded_list = chunk_embedder.embed_chunks(chunks_list)

    assert isinstance(embedded_list, list)
    # The empty chunk (index 1) should be skipped
    assert len(embedded_list) == 4

    # Check the IDs of the successful chunks
    assert embedded_list[0].chunk_id == 'chunk_0'
    assert embedded_list[1].chunk_id == 'chunk_2'
    assert embedded_list[2].chunk_id == 'chunk_3'
    assert embedded_list[3].chunk_id == 'chunk_4'

    # Verify content and metadata for a couple of chunks
    assert embedded_list[0].content == sample_chunk_data['content']
    assert embedded_list[0].metadata['link'] == sample_chunk_data['link']

    assert embedded_list[2].content == 'Only content'
    assert embedded_list[2].metadata['link'] == sample_chunk_data['link'] # Metadata should be from the original record


def test_embed_chunks_empty_list(chunk_embedder):
    chunk_embedder.load_model() # Ensure model is loaded
    embedded_list = chunk_embedder.embed_chunks([])
    assert isinstance(embedded_list, list)
    assert len(embedded_list) == 0

# --- Serialization Tests (save/load) ---

@pytest.fixture
def temp_files(tmp_path):
    """Fixture for temporary file paths."""
    embeddings_path = tmp_path / "temp_embeddings.json"
    return embeddings_path

def test_save_and_load_embeddings(chunk_embedder, sample_embedded_chunk, temp_files):
    # Create a list of sample embedded chunks
    embedded_list = [sample_embedded_chunk]

    # Save the embeddings
    chunk_embedder.save_embeddings(embedded_list, temp_files)

    # Check if the file was created
    assert temp_files.exists()

    # Load the embeddings back
    loaded_list = chunk_embedder.load_embeddings(temp_files)

    # Verify the loaded data
    assert isinstance(loaded_list, list)
    assert len(loaded_list) == 1
    loaded_chunk = loaded_list[0]

    assert isinstance(loaded_chunk, EmbeddedChunk)
    assert loaded_chunk.chunk_id == embedded_list[0].chunk_id
    assert loaded_chunk.title == embedded_list[0].title
    assert loaded_chunk.content == embedded_list[0].content
    assert isinstance(loaded_chunk.embedding, np.ndarray)
    assert loaded_chunk.embedding.shape == embedded_list[0].embedding.shape
    assert np.allclose(loaded_chunk.embedding, embedded_list[0].embedding) # Compare embeddings
    assert loaded_chunk.metadata == embedded_list[0].metadata

def test_save_embeddings_empty_list(chunk_embedder, temp_files):
    chunk_embedder.save_embeddings([], temp_files)
    assert temp_files.exists()
    with open(temp_files, 'r', encoding='utf-8') as f:
        data = json.load(f)
        assert data == [] # Should save an empty list

def test_load_embeddings_empty_file(chunk_embedder, temp_files):
    # Create an empty JSON file
    with open(temp_files, 'w', encoding='utf-8') as f:
        json.dump([], f)

    loaded_list = chunk_embedder.load_embeddings(temp_files)
    assert isinstance(loaded_list, list)
    assert len(loaded_list) == 0

def test_load_embeddings_file_not_found(chunk_embedder):
    non_existent_path = Path("non_existent_embeddings.json")
    with pytest.raises(FileNotFoundError):
         chunk_embedder.load_embeddings(non_existent_path)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])