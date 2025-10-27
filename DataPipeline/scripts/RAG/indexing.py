"""index.py - Create and manage FAISS index for document retrieval."""

import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np

from .embedding import ChunkEmbedder, EmbeddedChunk

# Setup logging
LOG_DIR = Path("/opt/airflow/logs")
LOG_DIR.mkdir(exist_ok=True)
log_file = LOG_DIR / f"indexing_{datetime.now().strftime('%Y-%m-%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class FAISSIndex:
    """FAISS index manager for document retrieval."""

    def __init__(self, dimension: int, index_type: str = 'flat'):
        """Initialize FAISS index.

        Args:
            dimension: Dimension of the embeddings
            index_type: Type of index ('flat', 'ivf', 'hnsw')
        """
        self.dimension = dimension
        self.index_type = index_type
        self.index: Optional[faiss.Index] = None
        self.chunks: List[EmbeddedChunk] = []
        logger.info(
            f"Initialized FAISSIndex with "
            f"dimension={dimension}, type={index_type}"
        )
    
    def create_index(self, nlist: int = 100) -> None:
        """Create FAISS index based on index_type.

        Args:
            nlist: Number of clusters for IVF index
        """
        try:
            logger.info(
                f"Creating {self.index_type} index "
                f"with dimension {self.dimension}"
            )
            
            if self.index_type == 'flat':
                # Flat L2 index - exact search, good for smaller datasets
                self.index = faiss.IndexFlatL2(self.dimension)
                logger.info("Created IndexFlatL2 for exact cosine similarity search")
                
            elif self.index_type == 'ivf':
                # IVF index - faster search for larger datasets
                quantizer = faiss.IndexFlatL2(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
                logger.info(f"Created IndexIVFFlat with {nlist} clusters")
                
            elif self.index_type == 'hnsw':
                # HNSW index - hierarchical navigable small world
                self.index = faiss.IndexHNSWFlat(self.dimension, 32)
                logger.info("Created IndexHNSWFlat for approximate nearest neighbor search")
                
            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")
            
            logger.info("Index created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create index: {str(e)}")
            raise
    
    def normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for cosine similarity.

        Note: BGE models already output normalized embeddings when
        normalize_embeddings=True is used during encoding.

        Args:
            embeddings: Array of embeddings

        Returns:
            Normalized embeddings
        """
        # Check if embeddings are already normalized
        norms = np.linalg.norm(embeddings, axis=1)
        if np.allclose(norms, 1.0, atol=1e-5):
            logger.info("Embeddings are already normalized (likely from BGE model)")
            return embeddings
        
        logger.info("Normalizing embeddings for cosine similarity")
        norms = norms.reshape(-1, 1)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return embeddings / norms
    
    def build_index(self, embedded_chunks: List[EmbeddedChunk]) -> None:
        """Build FAISS index from embedded chunks.

        Args:
            embedded_chunks: List of EmbeddedChunk objects
        """
        try:
            if not embedded_chunks:
                raise ValueError("No embedded chunks provided")
            
            logger.info(f"Building index from {len(embedded_chunks)} chunks")
            
            # Store chunks for later retrieval
            self.chunks = embedded_chunks
            
            # Extract embeddings
            embeddings = np.array([chunk.embedding for chunk in embedded_chunks])
            logger.info(f"Embeddings shape: {embeddings.shape}")
            
            # Normalize embeddings for cosine similarity
            embeddings = self.normalize_embeddings(embeddings)
            logger.info("Embeddings normalized for cosine similarity")
            
            # Ensure embeddings are contiguous and float32
            embeddings = np.ascontiguousarray(embeddings.astype('float32'))
            
            # Create index if not already created
            if self.index is None:
                self.create_index()
            
            # Train index if needed (for IVF)
            if self.index_type == 'ivf':
                logger.info("Training IVF index...")
                self.index.train(embeddings)
                logger.info("IVF index training complete")
            
            # Add embeddings to index
            logger.info("Adding embeddings to index...")
            self.index.add(embeddings)
            logger.info(f"Successfully added {self.index.ntotal} vectors to index")
            
        except Exception as e:
            logger.error(f"Failed to build index: {str(e)}")
            raise
    
    def save_index(self, index_path: Path, chunks_path: Path) -> None:
        """
        Save FAISS index and chunks to disk
        
        Args:
            index_path: Path to save FAISS index
            chunks_path: Path to save chunks metadata
        """
        try:
            if self.index is None:
                raise ValueError("No index to save. Build index first.")
            
            logger.info(f"Saving index to {index_path}")
            logger.info(f"Saving chunks to {chunks_path}")
            
            # Create directories if they don't exist
            index_path.parent.mkdir(parents=True, exist_ok=True)
            chunks_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, str(index_path))
            logger.info(f"FAISS index saved successfully")
            
            # Save chunks using pickle
            with open(chunks_path, 'wb') as f:
                pickle.dump(self.chunks, f)
            logger.info(f"Chunks saved successfully ({len(self.chunks)} items)")
            
        except Exception as e:
            logger.error(f"Failed to save index: {str(e)}")
            raise
    
    def load_index(self, index_path: Path, chunks_path: Path) -> None:
        """
        Load FAISS index and chunks from disk
        
        Args:
            index_path: Path to FAISS index file
            chunks_path: Path to chunks metadata file
        """
        try:
            logger.info(f"Loading index from {index_path}")
            logger.info(f"Loading chunks from {chunks_path}")
            
            # Load FAISS index
            self.index = faiss.read_index(str(index_path))
            logger.info(f"FAISS index loaded successfully ({self.index.ntotal} vectors)")
            
            # Load chunks
            with open(chunks_path, 'rb') as f:
                self.chunks = pickle.load(f)
            logger.info(f"Chunks loaded successfully ({len(self.chunks)} items)")
            
            # Verify consistency
            if len(self.chunks) != self.index.ntotal:
                logger.warning(
                    f"Mismatch: {len(self.chunks)} chunks but {self.index.ntotal} vectors in index"
                )
            
        except Exception as e:
            logger.error(f"Failed to load index: {str(e)}")
            raise
    
    def get_index_stats(self) -> dict:
        """
        Get statistics about the index
        
        Returns:
            Dictionary with index statistics
        """
        if self.index is None:
            return {"status": "not_initialized"}
        
        stats = {
            "status": "initialized",
            "index_type": self.index_type,
            "dimension": self.dimension,
            "total_vectors": self.index.ntotal,
            "total_chunks": len(self.chunks),
            "is_trained": getattr(self.index, 'is_trained', True)
        }
        
        return stats


def main():
    """Example usage - for standalone testing only."""
    INPUT_FILE = Path("/opt/airflow/data/RAG/index/embeddings.json")
    OUTPUT_FILE_INDEX = Path("/opt/airflow/data/RAG/index/index.bin")
    OUTPUT_FILE_DATA = Path("/opt/airflow/data/RAG/index/data.pkl")
    
    try:
        # Load embeddings
        logger.info("Loading embeddings...")
        embedded_chunks = ChunkEmbedder.load_embeddings(INPUT_FILE)
        
        if not embedded_chunks:
            logger.error("No embeddings found")
            return
        
        # Get embedding dimension
        embedding_dim = embedded_chunks[0].embedding.shape[0]
        logger.info(f"Embedding dimension: {embedding_dim}")
        
        # Create and build index
        logger.info("Creating FAISS index...")
        faiss_index = FAISSIndex(dimension=embedding_dim, index_type='flat')
        faiss_index.build_index(embedded_chunks)
        
        # Print stats
        stats = faiss_index.get_index_stats()
        logger.info(f"Index stats: {stats}")
        
        # Save index
        faiss_index.save_index(OUTPUT_FILE_INDEX, OUTPUT_FILE_DATA)
        
        # Verify by loading
        logger.info("Verifying by loading index...")
        verify_index = FAISSIndex(dimension=embedding_dim, index_type='flat')
        verify_index.load_index(OUTPUT_FILE_INDEX, OUTPUT_FILE_DATA)
        
        verify_stats = verify_index.get_index_stats()
        logger.info(f"Verification stats: {verify_stats}")
        
        logger.info("Index creation and verification complete!")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()