"""Embedding module for generating embeddings using sentence transformers."""

import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

# Setup logging
LOG_DIR = Path("/opt/airflow/logs")
LOG_DIR.mkdir(exist_ok=True)
log_file = LOG_DIR / f"embedding_{datetime.now().strftime('%Y-%m-%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class EmbeddedChunk:
    """Data class for a chunk with its embedding."""
    chunk_id: str
    title: str
    content: str
    embedding: np.ndarray
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        d['embedding'] = self.embedding.tolist()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmbeddedChunk':
        """Create from dictionary."""
        data['embedding'] = np.array(data['embedding'])
        return cls(**data)


class ChunkEmbedder:
    """Generate embeddings for document chunks."""

    def __init__(self, model_name: str = 'BAAI/llm-embedder'):
        """Initialize the embedder with a sentence transformer model.

        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        logger.info(f"Initializing ChunkEmbedder with model: {model_name}")
        
    def load_model(self) -> None:
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(
                f"Model loaded successfully. "
                f"Embedding dimension: {embedding_dim}"
            )
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {str(e)}")
            raise
    
    def create_combined_text(self, chunk: Dict[str, Any]) -> str:
        """
        Combine title and content for embedding
        
        Args:
            chunk: Dictionary containing chunk data
            
        Returns:
            Combined text string
        """
        title = chunk.get('title', '').strip()
        content = chunk.get('content', '').strip()
        
        # Weight title more heavily by including it twice
        combined = f"{title}. {title}. {content}"
        return combined
    
    def embed_chunk(self, chunk: Dict[str, Any], chunk_id: str) -> Optional[EmbeddedChunk]:
        """
        Generate embedding for a single chunk
        
        Args:
            chunk: Dictionary containing chunk data
            chunk_id: Unique identifier for the chunk
            
        Returns:
            EmbeddedChunk object or None if embedding fails
        """
        try:
            if self.model is None:
                raise ValueError("Model not loaded. Call load_model() first.")
            
            # Create combined text for embedding
            text = self.create_combined_text(chunk)
            
            if not text.strip():
                logger.warning(f"Empty text for chunk {chunk_id}, skipping")
                return None
            
            # Generate embedding
            embedding = self.model.encode(text, convert_to_numpy=True)

            # Check if embedding is empty or invalid
            if embedding.size == 0:
                logger.warning(f"Empty embedding generated for chunk {chunk_id}, skipping")
                return None
            
            # Extract metadata
            metadata = {
                k:v for k,v in chunk.items() if k not in ['content', 'title']
            }
            
            embedded_chunk = EmbeddedChunk(
                chunk_id=chunk_id,
                title=chunk.get('title', ''),
                content=chunk.get('content', ''),
                embedding=embedding,
                metadata=metadata
            )
            
            logger.debug(f"Successfully embedded chunk {chunk_id}")
            return embedded_chunk
            
        except Exception as e:
            logger.error(f"Failed to embed chunk {chunk_id}: {str(e)}")
            return None
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[EmbeddedChunk]:
        """
        Generate embeddings for multiple chunks
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            List of EmbeddedChunk objects
        """
        if self.model is None:
            self.load_model()
        
        logger.info(f"Starting embedding process for {len(chunks)} chunks")
        embedded_chunks = []
        failed_count = 0
        
        for idx, chunk in enumerate(chunks):
            chunk_id = f"chunk_{idx}"
            
            try:
                embedded = self.embed_chunk(chunk, chunk_id)
                if embedded:
                    embedded_chunks.append(embedded)
                else:
                    failed_count += 1
                    
                # Log progress every 100 chunks
                if (idx + 1) % 100 == 0:
                    logger.info(f"Processed {idx + 1}/{len(chunks)} chunks")
                    
            except Exception as e:
                logger.error(f"Error processing chunk {idx}: {str(e)}")
                failed_count += 1
        
        logger.info(f"Embedding complete. Successful: {len(embedded_chunks)}, Failed: {failed_count}")
        return embedded_chunks
    
    def save_embeddings(self, embedded_chunks: List[EmbeddedChunk], output_path: Path) -> None:
        """
        Save embedded chunks to a JSON file
        
        Args:
            embedded_chunks: List of EmbeddedChunk objects
            output_path: Path to output file
        """
        try:
            logger.info(f"Saving {len(embedded_chunks)} embeddings to {output_path}")
            
            # Convert to dictionaries
            data = [chunk.to_dict() for chunk in embedded_chunks]
            
            # Save to JSON
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Successfully saved embeddings to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save embeddings: {str(e)}")
            raise
    
    @staticmethod
    def load_embeddings(input_path: Path) -> List[EmbeddedChunk]:
        """
        Load embedded chunks from a JSON file
        
        Args:
            input_path: Path to input file
            
        Returns:
            List of EmbeddedChunk objects
        """
        try:
            logger.info(f"Loading embeddings from {input_path}")
            
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            embedded_chunks = [EmbeddedChunk.from_dict(d) for d in data]
            logger.info(f"Successfully loaded {len(embedded_chunks)} embeddings")
            return embedded_chunks
            
        except Exception as e:
            logger.error(f"Failed to load embeddings: {str(e)}")
            raise


def main():
    """Main execution function - for standalone testing only."""
    INPUT_FILE = Path("/opt/airflow/data/RAG/chunked_data/chunks.json")
    OUTPUT_FILE = Path("/opt/airflow/data/RAG/index/embeddings.json")
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        chunk_file = json.load(f)

    chunks = [record for record in chunk_file]
    
    # Initialize embedder
    embedder = ChunkEmbedder(model_name='BAAI/llm-embedder')
    
    # Embed chunks
    embedded_chunks = embedder.embed_chunks(chunks)
    
    # Save embeddings
    embedder.save_embeddings(embedded_chunks, OUTPUT_FILE)
    
    # Load embeddings (for verification)
    loaded_chunks = ChunkEmbedder.load_embeddings(OUTPUT_FILE)
    logger.info(f"Verification: Loaded {len(loaded_chunks)} chunks")


if __name__ == "__main__":
    main()