import os
import pickle
import sys
from pathlib import Path

import faiss
import numpy as np
import pytest

# Adjust path if necessary
current_dir = os.path.dirname(os.path.abspath(__file__))
data_pipeline_dir = os.path.dirname(current_dir)
sys.path.insert(0, data_pipeline_dir)

from scripts.RAG.embedding import EmbeddedChunk
from scripts.RAG.indexing import FAISSIndex

# Fixture for a FAISSIndex instance
@pytest.fixture
def faiss_index_instance(dimension=768, index_type='flat'):
    return FAISSIndex(dimension=dimension, index_type=index_type)

# Fixture to generate dummy EmbeddedChunk objects
@pytest.fixture
def dummy_embedded_chunks(dimension=768, num_chunks=10):
    chunks = []
    for i in range(num_chunks):
        embedding = np.random.rand(dimension).astype('float32')
        # Normalize embeddings to simulate BGE output
        embedding = embedding / np.linalg.norm(embedding)
        chunks.append(EmbeddedChunk(
            chunk_id=f'chunk_{i}',
            title=f'Title {i}',
            content=f'Content of chunk {i}.',
            embedding=embedding,
            metadata={'source': 'test', 'index': i}
        ))
    return chunks

# --- Index Creation Tests ---

def test_index_creation_flat(faiss_index_instance):
    faiss_index_instance.create_index()
    assert isinstance(faiss_index_instance.index, faiss.IndexFlatL2)
    assert faiss_index_instance.index.d == 768

def test_index_creation_ivf(faiss_index_instance):
    ivf_index = FAISSIndex(dimension=768, index_type='ivf')
    ivf_index.create_index(nlist=50)
    assert isinstance(ivf_index.index, faiss.IndexIVFFlat)
    assert ivf_index.index.d == 768
    assert ivf_index.index.nlist == 50
    assert not ivf_index.index.is_trained # IVF index needs training

def test_index_creation_hnsw(faiss_index_instance):
    hnsw_index = FAISSIndex(dimension=768, index_type='hnsw')
    hnsw_index.create_index()
    assert isinstance(hnsw_index.index, faiss.IndexHNSWFlat)
    assert hnsw_index.index.d == 768
    assert hnsw_index.index.is_trained # HNSWFlat is immediately trained

def test_index_creation_unsupported_type():
    with pytest.raises(ValueError, match="Unsupported index type: invalid"):
        invalid_index = FAISSIndex(dimension=768, index_type='invalid')
        invalid_index.create_index()

# --- Embedding Normalization Tests ---

def test_normalize_embeddings_already_normalized(faiss_index_instance):
    embeddings = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype='float32')
    normalized = faiss_index_instance.normalize_embeddings(embeddings)
    assert np.allclose(normalized, embeddings) # Should return the same array if already normalized

def test_normalize_embeddings_unnormalized(faiss_index_instance):
    embeddings = np.array([[1.0, 1.0], [2.0, 0.0]], dtype='float32')
    normalized = faiss_index_instance.normalize_embeddings(embeddings)

    assert np.allclose(np.linalg.norm(normalized, axis=1), 1.0) # Check if norms are 1
    # Check direction
    assert np.allclose(normalized[0], embeddings[0] / np.linalg.norm(embeddings[0]))
    assert np.allclose(normalized[1], embeddings[1] / np.linalg.norm(embeddings[1]))

# --- Index Building Tests ---

def test_build_index_success(faiss_index_instance, dummy_embedded_chunks):
    faiss_index_instance.build_index(dummy_embedded_chunks)
    assert faiss_index_instance.index is not None
    assert faiss_index_instance.index.ntotal == len(dummy_embedded_chunks)
    assert faiss_index_instance.index.is_trained # Index should be trained after build

def test_build_index_empty_chunks(faiss_index_instance):
    with pytest.raises(ValueError, match="No embedded chunks provided"):
        faiss_index_instance.build_index([])

# --- Save and Load Tests ---

def test_save_and_load_index(faiss_index_instance, dummy_embedded_chunks, tmp_path):
    faiss_index_instance.build_index(dummy_embedded_chunks)

    index_path = tmp_path / "test_index.bin"
    chunks_path = tmp_path / "test_chunks.pkl"

    faiss_index_instance.save_index(index_path, chunks_path)

    assert index_path.exists()
    assert chunks_path.exists()

    loaded_index = FAISSIndex(dimension=faiss_index_instance.dimension, index_type=faiss_index_instance.index_type)
    loaded_index.load_index(index_path, chunks_path)

    assert loaded_index.index is not None
    assert loaded_index.chunks is not None
    assert loaded_index.index.ntotal == faiss_index_instance.index.ntotal
    assert len(loaded_index.chunks) == len(faiss_index_instance.chunks)

    # Optional: Deep compare loaded chunks with original chunks
    for i in range(len(dummy_embedded_chunks)):
        assert loaded_index.chunks[i].chunk_id == dummy_embedded_chunks[i].chunk_id
        assert loaded_index.chunks[i].title == dummy_embedded_chunks[i].title
        assert loaded_index.chunks[i].content == dummy_embedded_chunks[i].content
        assert np.allclose(loaded_index.chunks[i].embedding, dummy_embedded_chunks[i].embedding)
        assert loaded_index.chunks[i].metadata == dummy_embedded_chunks[i].metadata


def test_save_index_not_built(faiss_index_instance, tmp_path):
    index_path = tmp_path / "test_index.bin"
    chunks_path = tmp_path / "test_chunks.pkl"
    with pytest.raises(ValueError, match="No index to save. Build index first."):
        faiss_index_instance.save_index(index_path, chunks_path)

def test_load_index_file_not_found(faiss_index_instance, tmp_path):
    non_existent_index_path = tmp_path / "non_existent_index.bin"
    non_existent_chunks_path = tmp_path / "non_existent_chunks.pkl"

    with pytest.raises(RuntimeError):  # FAISS raises RuntimeError for missing files
        faiss_index_instance.load_index(non_existent_index_path, non_existent_chunks_path)

    # Test case where only index file is missing
    chunks_path = tmp_path / "temp_chunks.pkl"
    with open(chunks_path, 'wb') as f:
        pickle.dump([], f)
    with pytest.raises(RuntimeError): # FAISS read_index raises RuntimeError for missing file
         faiss_index_instance.load_index(non_existent_index_path, chunks_path)

    # Test case where only chunks file is missing
    index_path = tmp_path / "temp_index.bin"
    temp_index = faiss.IndexFlatL2(768)
    faiss.write_index(temp_index, str(index_path))
    with pytest.raises(FileNotFoundError):
        faiss_index_instance.load_index(index_path, non_existent_chunks_path)

# --- Index Stats Tests ---

def test_get_index_stats(faiss_index_instance, dummy_embedded_chunks):
    # Stats before building
    stats_before = faiss_index_instance.get_index_stats()
    assert stats_before == {"status": "not_initialized"}

    # Build the index
    faiss_index_instance.build_index(dummy_embedded_chunks)

    # Stats after building
    stats_after = faiss_index_instance.get_index_stats()
    assert stats_after["status"] == "initialized"
    assert stats_after["index_type"] == "flat"
    assert stats_after["dimension"] == 768
    assert stats_after["total_vectors"] == len(dummy_embedded_chunks)
    assert stats_after["total_chunks"] == len(dummy_embedded_chunks)
    assert stats_after["is_trained"] is True

def test_get_index_stats_ivf(dummy_embedded_chunks):
    ivf_index_instance = FAISSIndex(dimension=768, index_type='ivf')
    ivf_index_instance.create_index(nlist=5)  # Reduced to 5 clusters (10 chunks >= 5 clusters)
    stats_created = ivf_index_instance.get_index_stats()
    assert stats_created["status"] == "initialized"
    assert stats_created["index_type"] == "ivf"
    assert stats_created["is_trained"] is False

    ivf_index_instance.build_index(dummy_embedded_chunks)
    stats_built = ivf_index_instance.get_index_stats()
    assert stats_built["is_trained"] is True

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])