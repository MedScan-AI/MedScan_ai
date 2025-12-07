"""
retrieval.py - Document retrieval with complete metadata
FIXED: Added all missing imports
"""

import json
import logging
import pickle
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

# NumPy for arrays
import numpy as np

# PyTorch for tensor operations
import torch

# FAISS for vector search
import faiss

# Transformers for embedding models
from transformers import (
    AutoModel, 
    AutoTokenizer
)

# Global cache for embedding model
_EMBEDDING_MODEL = None
_EMBEDDING_TOKENIZER = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_embedding_model(model_name: str = "BAAI/llm-embedder"):
    """Load and return the embedding model & tokenizer (cached)"""
    global _EMBEDDING_MODEL, _EMBEDDING_TOKENIZER
    if _EMBEDDING_MODEL is None:
        logger.info("Loading embedding model: %s", model_name)
        _EMBEDDING_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        _EMBEDDING_MODEL = AutoModel.from_pretrained(model_name)
        if torch.cuda.is_available():
            _EMBEDDING_MODEL = _EMBEDDING_MODEL.to("cuda")
    return _EMBEDDING_MODEL, _EMBEDDING_TOKENIZER


def get_embeddings(text: str) -> np.ndarray:
    """
    Compute dense embedding for text
    
    Args:
        text: Input text to embed
        
    Returns:
        1D numpy array of embedding (shape: 768)
    """
    model, tokenizer = get_embedding_model()
    
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    
    # Move to GPU if available
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    # Generate embedding
    with torch.no_grad():
        outputs = model(**inputs)
        
        # Extract CLS token or pooled output
        if hasattr(outputs, "last_hidden_state"):
            emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        elif hasattr(outputs, "pooler_output"):
            emb = outputs.pooler_output.cpu().numpy()
        else:
            emb = outputs[0][:, 0, :].cpu().numpy()
    
    return emb.reshape(-1)


@dataclass
class DocumentRecord:
    """Complete document record with content and metadata"""
    chunk_id: str
    title: str
    content: str
    embedding: np.ndarray
    metadata: Dict[str, Any]
    
    def get_metadata_str(self) -> str:
        """Format metadata for display in prompts"""
        parts = []
        
        # Source
        if 'source' in self.metadata and self.metadata['source'] not in ['Unknown', 'N/A', '']:
            parts.append(self.metadata['source'])
        
        # Country
        if 'country' in self.metadata and self.metadata['country'] not in ['Unknown', 'N/A', '']:
            parts.append(f"Country: {self.metadata['country']}")
        
        # Date
        if 'publish_date' in self.metadata:
            date_str = clean_date(self.metadata['publish_date'])
            if date_str:
                parts.append(f"Published: {date_str}")
        
        return " | ".join(parts) if parts else "Source: Internal Document"


def clean_date(date_input: Any) -> Optional[str]:
    """
    Convert various date formats to YYYY-MM-DD or best available
    
    Args:
        date_input: Date in various formats
        
    Returns:
        Cleaned date string or None
    """
    if not date_input or date_input in ['Unknown', 'N/A', '']:
        return None
    
    # Try parsing common formats
    formats = ['%Y-%m-%d', '%Y-%m', '%Y', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d']
    
    for fmt in formats:
        try:
            parsed = datetime.strptime(str(date_input), fmt)
            if fmt == '%Y':
                return parsed.strftime('%Y')
            elif fmt == '%Y-%m':
                return parsed.strftime('%Y-%m')
            else:
                return parsed.strftime('%Y-%m-%d')
        except:
            continue
    
    # Try extracting year with regex
    year_match = re.search(r'(19|20)\d{2}', str(date_input))
    if year_match:
        return year_match.group(0)
    
    return None


class DocumentRetriever:
    """Retrieve documents with complete metadata using FAISS index"""
    
    def __init__(
        self,
        index_path: Path,
        chunks_path: Path,
        original_data_path: Optional[Path] = None
    ):
        """
        Initialize retriever
        
        Args:
            index_path: Path to FAISS index file
            chunks_path: Path to pickled chunks file
            original_data_path: Path to original JSON data (for metadata enrichment)
        """
        self.index_path = index_path
        self.chunks_path = chunks_path
        self.original_data_path = original_data_path
        
        self.index: Optional[faiss.Index] = None
        self.documents: List[DocumentRecord] = []
        self.original_data: Dict[str, Any] = {}
        
        self._load_index()
        if original_data_path and original_data_path.exists():
            self._load_original_data()
    
    def _load_index(self) -> None:
        """Load FAISS index and document chunks"""
        try:
            logger.info(f"Loading FAISS index from {self.index_path}")
            self.index = faiss.read_index(str(self.index_path))
            logger.info(f"Index loaded: {self.index.ntotal} vectors")
            
            logger.info(f"Loading chunks from {self.chunks_path}")
            with open(self.chunks_path, 'rb') as f:
                chunks = pickle.load(f)
            
            # Convert to DocumentRecord
            self.documents = []
            for chunk in chunks:
                doc_record = DocumentRecord(
                    chunk_id=getattr(chunk, 'chunk_id', ''),
                    title=getattr(chunk, 'title', ''),
                    content=getattr(chunk, 'content', ''),
                    embedding=getattr(chunk, 'embedding', np.array([])),
                    metadata=getattr(chunk, 'metadata', {})
                )
                self.documents.append(doc_record)
            
            logger.info(f"Loaded {len(self.documents)} document records")
            
        except Exception as e:
            logger.error(f"Failed to load index: {str(e)}")
            raise
    
    def _load_original_data(self) -> None:
        """Load original data for metadata enrichment"""
        try:
            logger.info(f"Loading original data from {self.original_data_path}")
            
            # Check if it's JSONL (multiple JSON objects, one per line)
            if str(self.original_data_path).endswith('.jsonl'):
                data = []
                with open(self.original_data_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                data.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
            else:
                # Regular JSON file
                with open(self.original_data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            
            # Index by document ID or title
            for item in data:
                key = item.get('id') or item.get('title')
                if key:
                    self.original_data[key] = item
            
            logger.info(f"Loaded {len(self.original_data)} original records")
            
        except Exception as e:
            logger.warning(f"Could not load original data: {str(e)}")
    
    def _enrich_metadata(self, doc: DocumentRecord) -> DocumentRecord:
        """Enrich document metadata from original data if available"""
        if not self.original_data:
            return doc
        
        # Try to find matching original record
        original = self.original_data.get(doc.title) or self.original_data.get(doc.chunk_id)
        
        if original:
            # Merge metadata
            enriched_metadata = {**doc.metadata}
            for key in ['source', 'country', 'publish_date', 'author', 'doi']:
                if key in original and original[key]:
                    enriched_metadata[key] = original[key]
            
            doc.metadata = enriched_metadata
        
        return doc
    
    def retrieve(
        self,
        query_embedding: np.ndarray,
        num_docs: int = 5,
        threshold: float = 0.0
    ) -> List[DocumentRecord]:
        """
        Retrieve top-k documents for query embedding
        
        Args:
            query_embedding: Query embedding vector
            num_docs: Number of documents to retrieve
            threshold: Minimum similarity threshold
            
        Returns:
            List of DocumentRecord objects with complete metadata
        """
        try:
            if self.index is None:
                raise ValueError("Index not loaded")
            
            # Normalize query embedding for cosine similarity
            query_embedding = query_embedding.reshape(1, -1).astype('float32')
            faiss.normalize_L2(query_embedding)
            
            # Search
            distances, indices = self.index.search(query_embedding, num_docs)
            
            # Get documents
            retrieved_docs = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    
                    # Calculate similarity (cosine similarity = 1 - L2_distance^2/2 for normalized vectors)
                    similarity = 1 - (distance ** 2) / 2
                    
                    if similarity >= threshold:
                        # Enrich metadata
                        doc = self._enrich_metadata(doc)
                        retrieved_docs.append(doc)
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents")
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"Retrieval error: {str(e)}")
            return []
    
    def format_context_for_prompt(
        self,
        documents: List[DocumentRecord],
        include_metadata: bool = True
    ) -> str:
        """
        Format retrieved documents for prompt
        
        Args:
            documents: List of retrieved documents
            include_metadata: Whether to include metadata
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for idx, doc in enumerate(documents, 1):
            if include_metadata:
                metadata_str = doc.get_metadata_str()
                header = f"[Source {idx}: {metadata_str}]"
            else:
                header = f"[Source {idx}]"
            
            # Combine title and content
            if doc.title:
                doc_text = f"{doc.title}\n{doc.content}"
            else:
                doc_text = doc.content
            
            context_parts.append(f"{header}\n{doc_text}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics"""
        return {
            "total_documents": len(self.documents),
            "index_vectors": self.index.ntotal if self.index else 0,
            "has_original_data": len(self.original_data) > 0,
            "original_records": len(self.original_data)
        }