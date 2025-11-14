"""
RAG_inference.py - Updated to read directly from GCS with local fallback
Supports running in Airflow, Cloud Build, and local development
"""
import logging
import json
import pickle
import gc
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss
import torch
from transformers import AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from google.cloud import storage
import tempfile

import os 
import sys
cur_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(cur_dir)
sys.path.insert(0, parent_dir)

import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

from models.models import ModelFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== GCS- Data Loading ====================

class RAGDataLoader:
    """Load RAG data from GCS with local fallback"""
    
    def __init__(
        self,
        bucket_name: str = "medscan-data",
        project_id: str = "medscanai-476203"
    ):
        """
        Initialize data loader.
        
        Args:
            bucket_name: GCS bucket name
            project_id: GCP project ID
        """
        self.bucket_name = bucket_name
        self.project_id = project_id
        
        # Try to initialize GCS client
        try:
            self.storage_client = storage.Client(project=project_id)
            self.bucket = self.storage_client.bucket(bucket_name)
            self.gcs_available = True
            logger.info(f"GCS client initialized (bucket: {bucket_name})")
        except Exception as e:
            logger.warning(f"⚠️  GCS not available: {e}")
            logger.warning("Will use local paths as fallback")
            self.gcs_available = False
    
    def _get_gcs_path(self, data_type: str) -> str:
        """Get GCS path for data type"""
        paths = {
            'embeddings': 'RAG/index/embeddings.json',
            'index': 'RAG/index/index.bin',
            'data': 'RAG/index/data.pkl'
        }
        return paths.get(data_type, '')
    
    def _get_local_fallback(self, data_type: str) -> Path:
        """Get local fallback path - tries multiple locations"""
        files = {
            'embeddings': 'embeddings.json',
            'index': 'index.bin',
            'data': 'data.pkl'
        }
        
        filename = files.get(data_type, '')
        
        # Try multiple base paths in order
        possible_bases = [
            Path('/opt/airflow/DataPipeline/data/RAG/index'),  # Airflow
            Path('/workspace/data/RAG/index'),  # Cloud Build
            Path(__file__).parent.parent.parent.parent / 'DataPipeline' / 'data' / 'RAG' / 'index',  # Local dev
        ]
        
        # Try each base path
        for base in possible_bases:
            full_path = base / filename
            if full_path.exists():
                return full_path
        
        # Return first option as default (will fail later if doesn't exist)
        return possible_bases[0] / filename
    
    def load_embeddings(self) -> Optional[List[Dict[str, Any]]]:
        """
        Load embeddings from GCS or local fallback.
        
        Returns:
            List of embedding records or None
        """
        try:
            # Try GCS first
            if self.gcs_available:
                gcs_path = self._get_gcs_path('embeddings')
                logger.info(f"Loading embeddings from GCS: gs://{self.bucket_name}/{gcs_path}")
                
                blob = self.bucket.blob(gcs_path)
                if blob.exists():
                    content = blob.download_as_text()
                    data = json.loads(content)
                    logger.info(f"Loaded {len(data)} embeddings from GCS")
                    return data
                else:
                    logger.warning(f"Embeddings not found in GCS: {gcs_path}")
            
            # Fallback to local
            local_path = self._get_local_fallback('embeddings')
            logger.info(f"Loading embeddings from local: {local_path}")
            
            if not local_path.exists():
                logger.error(f"Embeddings not found: {local_path}")
                return None
            
            with open(local_path, 'r') as f:
                data = json.load(f)
            
            logger.info(f"Loaded {len(data)} embeddings from local")
            return data
            
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            return None
    
    def load_faiss_index(self) -> Optional[faiss.Index]:
        """
        Load FAISS index from GCS or local fallback.
        
        Returns:
            FAISS index or None
        """
        try:
            # Try GCS first
            if self.gcs_available:
                gcs_path = self._get_gcs_path('index')
                logger.info(f"Loading FAISS index from GCS: gs://{self.bucket_name}/{gcs_path}")
                
                blob = self.bucket.blob(gcs_path)
                if blob.exists():
                    # Download to temp file (FAISS needs file path)
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as tmp:
                        blob.download_to_filename(tmp.name)
                        index = faiss.read_index(tmp.name)
                        os.unlink(tmp.name)  # Clean up temp file
                    
                    logger.info(f"Loaded FAISS index from GCS ({index.ntotal} vectors)")
                    return index
                else:
                    logger.warning(f"Index not found in GCS: {gcs_path}")
            
            # Fallback to local
            local_path = self._get_local_fallback('index')
            logger.info(f"Loading FAISS index from local: {local_path}")
            
            if not local_path.exists():
                logger.error(f"Index not found: {local_path}")
                return None
            
            index = faiss.read_index(str(local_path))
            logger.info(f"Loaded FAISS index from local ({index.ntotal} vectors)")
            return index
            
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            return None
    
    def load_data_pkl(self) -> Optional[Any]:
        """
        Load pickled data from GCS or local fallback.
        
        Returns:
            Unpickled data or None
        """
        try:
            # Try GCS first
            if self.gcs_available:
                gcs_path = self._get_gcs_path('data')
                logger.info(f"Loading data.pkl from GCS: gs://{self.bucket_name}/{gcs_path}")
                
                blob = self.bucket.blob(gcs_path)
                if blob.exists():
                    content = blob.download_as_bytes()
                    data = pickle.loads(content)
                    logger.info(f"Loaded data.pkl from GCS")
                    return data
                else:
                    logger.warning(f"data.pkl not found in GCS: {gcs_path}")
            
            # Fallback to local
            local_path = self._get_local_fallback('data')
            logger.info(f"Loading data.pkl from local: {local_path}")
            
            if not local_path.exists():
                logger.error(f"data.pkl not found: {local_path}")
                return None
            
            with open(local_path, 'rb') as f:
                data = pickle.load(f)
            
            logger.info(f"Loaded data.pkl from local")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data.pkl: {e}")
            return None


# Create global loader instance
_data_loader = RAGDataLoader()


# ==================== Wrapper Functions (Backward Compatibility) ====================

def load_embeddings_data() -> Optional[List[Dict[str, Any]]]:
    """Load embeddings data (wrapper for backward compatibility)"""
    return _data_loader.load_embeddings()


def load_faiss_index() -> Optional[faiss.Index]:
    """Load FAISS index (wrapper for backward compatibility)"""
    return _data_loader.load_faiss_index()


def load_data_pkl() -> Optional[Any]:
    """Load pickled data (wrapper for backward compatibility)"""
    return _data_loader.load_data_pkl()


# ==================== Config Loading with GCS Support ====================

def load_config(config_path: str = "utils/RAG_config.json") -> Dict[str, Any]:
    """
    Load configuration from GCS or local file.
    
    Args:
        config_path: Path to config file (can be GCS URI or local path)
        
    Returns:
        Dictionary containing configuration parameters
    """
    try:
        # Check if it's a GCS URI
        if config_path.startswith('gs://'):
            logger.info(f"Loading config from GCS: {config_path}")
            
            # Parse GCS URI
            parts = config_path.replace('gs://', '').split('/', 1)
            bucket_name = parts[0]
            blob_path = parts[1] if len(parts) > 1 else ''
            
            # Download from GCS
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            
            content = blob.download_as_text()
            config = json.loads(content)
            logger.info("Config loaded from GCS")
            return config
        
        # Local file - try multiple locations
        logger.info(f"Loading config from local: {config_path}")
        
        possible_paths = [
            Path(config_path),
            Path(__file__).parent.parent / 'utils' / 'RAG_config.json',
            Path('/workspace/RAG_config.json'),  # Cloud Build
            Path(__file__).parent.parent.parent.parent / 'ModelDevelopment' / 'RAG' / 'utils' / 'RAG_config.json',  # Absolute
        ]
        
        for path in possible_paths:
            if path.exists():
                with open(path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Config loaded from {path}")
                return config
        
        # If none found, raise error
        raise FileNotFoundError(f"Config not found in any location. Tried: {[str(p) for p in possible_paths]}")
        
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise


# ==================== Core RAG Functions (Unchanged) ====================

def get_embedding(query: str, model_name: str = "BAAI/llm-embedder") -> Optional[np.ndarray]:
    """
    Generate embedding vector for query using sentence transformer.
    
    Args:
        query: Input text to embed
        model_name: Name of the embedding model
        
    Returns:
        Numpy array of embedding vector or None on failure
    """
    try:
        logger.info(f"Generating embedding with model: {model_name}")
        
        model = SentenceTransformer(model_name, device='cpu')
        embedding = model.encode(query, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        logger.info(f"Embedding generated with shape: {embedding.shape}")
        del model
        gc.collect()
        
        return embedding
        
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None


def retrieve_documents(
    embedding: np.ndarray,
    index: faiss.Index,
    embeddings_data: List[Dict[str, Any]],
    k: int,
    retrieval_method: str = "similarity"
) -> Optional[List[Dict[str, Any]]]:
    """
    Retrieve top-k documents from FAISS index.
    
    Args:
        embedding: Query embedding vector
        index: FAISS index containing document embeddings
        embeddings_data: List of embedding records with metadata
        k: Number of documents to retrieve
        retrieval_method: Method for retrieval (e.g., 'similarity')
        
    Returns:
        List of retrieved documents with full content and metadata or None on failure
    """
    try:
        logger.info(f"Retrieving {k} documents using {retrieval_method} method")
        
        # Reshape embedding for FAISS (needs 2D array)
        query_vector = embedding.reshape(1, -1).astype('float32')
        
        # Search FAISS index
        distances, indices = index.search(query_vector, k)
        
        # Format results with full document data
        documents = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx != -1 and idx < len(embeddings_data):  # Valid index
                record = embeddings_data[idx]
                documents.append({
                    "rank": i + 1,
                    "doc_id": idx,
                    "chunk_id": record.get("chunk_id"),
                    "title": record.get("title"),
                    "content": record.get("content"),
                    "metadata": record.get("metadata", {}),
                    "distance": float(dist),
                    "score": float(1 / (1 + dist))  # Convert distance to similarity score
                })
        
        logger.info(f"Retrieved {len(documents)} documents")
        return documents
        
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        return None


def generate_response(
    query: str,
    documents: List[Dict[str, Any]],
    config: Dict[str, Any]
) -> Optional[Tuple[str, int, int]]:
    """
    Generate response using LLM with retrieved context.
    
    Args:
        query: Original user query
        documents: Retrieved documents with metadata
        config: Configuration containing model params and prompt
        
    Returns:
        Tuple of (response string, input_tokens, output_tokens) or None on failure
    """
    try:
        logger.info("Generating response")
        
        # Extract config parameters
        model_name = config.get("model_name", None)
        model_type = config.get("model_type", None)
        prompt_template = config.get("prompt", None)

        if model_name is None or model_type is None or prompt_template is None:
            raise ValueError(f"Empty/None values detected\n\nModel Name: {model_name}\nModel type: {model_type}\nPrompt:{prompt_template}")

        temperature = config.get("temperature", 0.7)
        top_p = config.get("top_p", 0.9)
        
        # Format context from documents with actual content
        context_parts = []
        for d in documents:
            title = d.get('title', 'Unknown')
            content = d.get('content', '')
            context_parts.append(f"Document {d['rank']} - {title}:\n{content}")
        
        context = "\n\n".join(context_parts)
        
        # Format prompt
        formatted_prompt = prompt_template.format(context=context, query=query)
        
        logger.info(f"Using model: {model_name} (type: {model_type})")
        logger.info(f"Temperature: {temperature}, Top-p: {top_p}")
        
        model = ModelFactory.create_model(model_name, temperature, top_p)
        response_d = model.infer(formatted_prompt)

        if response_d is None or response_d.get('success') != True:
            raise ValueError(f"Error generating response - {response_d}")

        response = response_d.get('generated_text', None)
        in_tokens = response_d.get('input_tokens', 0)
        out_tokens = response_d.get('output_tokens', 0)
        
        # Add references section with links
        references = "\n\n**References:**\n"
        for d in documents:
            link = d.get('metadata', {}).get('link', '')
            title = d.get('title', 'Unknown')
            if link:
                references += f"{d['rank']}. [{title}]({link})\n"
            else:
                references += f"{d['rank']}. {title}\n"
        
        response += references

        logger.info("Response generated successfully")
        return response, in_tokens, out_tokens
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return None


def compute_hallucination_score(pairs: tuple, context: str, model) -> Optional[float]:
    """
    Compute hallucination score for response given context.
    
    Args:
        pairs: Tuple of (response, context) pairs
        context: Source context text
        model: Hallucination evaluation model
        
    Returns:
        Hallucination score or None on failure
    """
    try:
        with torch.no_grad():
            scores = model.predict(pairs)
            # Convert to Python list if it's a tensor
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().tolist()
            return scores[0] if isinstance(scores, list) else float(scores)
    except Exception as e:
        logger.error(f"Error during hallucination evaluation: {str(e)}")
        return None


def compute_stats(
    query: str,
    response: str,
    retrieved_docs: List[Dict[str, Any]],
    config: Dict[str, Any],
    prompt_template: str,
    in_tokens: int = 0,
    out_tokens: int = 0
) -> Dict[str, Any]:
    """
    Compute statistics and metrics for the RAG pipeline response.
    
    Args:
        query: Original user query
        response: Generated response
        retrieved_docs: List of retrieved documents with metadata
        config: Configuration dict with hyperparameters
        prompt_template: The formatted prompt used for generation
        in_tokens: Number of input tokens
        out_tokens: Number of output tokens
        
    Returns:
        Dictionary containing all computed statistics
    """
    try:
        logger.info("Computing response statistics")
        
        # 1. Calculate average retrieval score
        if retrieved_docs:
            scores = [doc.get("score", 0.0) for doc in retrieved_docs]
            avg_retrieval_score = sum(scores) / len(scores)
        else:
            avg_retrieval_score = 0.0
        logger.info(f"Average retrieval score: {avg_retrieval_score:.4f}")
        
        # 2. Compute hallucination scores per source
        hallucination_scores = {}
        if retrieved_docs:
            scores_list = []
            try:
                hallucination_model = AutoModelForSequenceClassification.from_pretrained(
                    "vectara/hallucination_evaluation_model", 
                    trust_remote_code=True
                )
                pairs = []
                for doc in retrieved_docs:
                    content = doc.get("content", "")
                    pairs.append((response, content))
                scores_list = compute_hallucination_score(pairs, content, hallucination_model)
                # Calculate average hallucination score
                if scores_list:
                    hallucination_scores["avg"] = round(sum(scores_list) / len(scores_list), 4)
                else:
                    hallucination_scores["avg"] = 0.0
            except Exception as e:
                logger.warning(f"Could not compute hallucination scores: {e}")
                hallucination_scores["avg"] = 0.0
        else:
            hallucination_scores["avg"] = 0.0
        
        logger.info(f"Hallucination scores computed: avg = {hallucination_scores.get('avg', 0.0):.4f}")
        
        # 3. Extract sampling hyperparameters
        sampling_params = {
            "num_docs": config.get("k", 5),
            "temperature": config.get("temperature", 0.7),
            "top_p": config.get("top_p", 0.9)
        }
        
        # 4 & 5. Include query and prompt
        stats = {
            "query": query,
            "prompt": prompt_template,
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "total_tokens": in_tokens + out_tokens,
            "avg_retrieval_score": round(avg_retrieval_score, 4),
            "hallucination_scores": hallucination_scores,
            "sampling_params": sampling_params,
            "num_retrieved_docs": len(retrieved_docs),
            "retrieved_doc_ids": [doc.get("chunk_id") for doc in retrieved_docs]
        }
        
        logger.info("Statistics computed successfully")
        return stats
        
    except Exception as e:
        logger.error(f"Error computing statistics: {e}")
        return {
            "query": query,
            "error": str(e),
            "input_tokens": 0,
            "output_tokens": 0,
            "avg_retrieval_score": 0.0,
            "hallucination_scores": {"avg": 0.0},
            "sampling_params": {}
        }


def run_rag_pipeline(
    query: str, 
    guardrail_checker: Optional[Any] = None
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Main RAG pipeline orchestrating all functions.
    
    Args:
        query: User query string
        guardrail_checker: Optional guardrail checker instance
        
    Returns:
        Tuple of (final_response, stats) or (None, None) if any step fails
    """
    try:
        # Load all required data
        config = load_config()
        embeddings_data = load_embeddings_data()
        index = load_faiss_index()
        
        if embeddings_data is None or index is None:
            return "Error loading required data files.", None
        
        # Get embedding
        embedding = get_embedding(query, config.get("embedding_model", "BAAI/llm-embedder"))
        if embedding is None:
            return "Error generating query embedding.", None
        
        # Retrieve documents
        k = config.get("k", 5)
        retrieval_method = config.get("retrieval_method", "similarity")
        documents = retrieve_documents(embedding, index, embeddings_data, k, retrieval_method)
        if not documents:
            return "Error retrieving documents.", None
        
        # Generate response
        result = generate_response(query, documents, config)
        if result is None:
            return "Error generating response.", None
        
        response, in_tokens, out_tokens = result
        
        # Add medical disclaimer footer
        footer = (
            "\n\n---\n"
            "**Important:** This information is for educational purposes only and "
            "should not replace professional medical advice. Please consult a "
            "healthcare provider for diagnosis, treatment, or medical guidance."
        )
        final_response = response + footer
        
        # Compute stats
        prompt_template = config.get("prompt", "")
        stats = compute_stats(
            query, 
            final_response, 
            documents, 
            config, 
            prompt_template,
            in_tokens,
            out_tokens
        )
        
        return final_response, stats
        
    except Exception as e:
        logger.error(f"Error in RAG pipeline: {e}")
        return "An error occurred processing your query.", None


if __name__ == "__main__":
    query = "What is Tuberculosis (TB)?"
    response, stats = run_rag_pipeline(query)

    logger.info(response)
    logger.info("*"*80)
    logger.info(stats)