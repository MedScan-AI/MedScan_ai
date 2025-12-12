"""
RAG_serve.py - Enhanced version with robust config normalization
Handles MLflow-generated configs without modification
"""
import os
import sys
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from fastapi import FastAPI, BackgroundTasks
import uvicorn
import torch
import numpy as np
from google.cloud import logging as cloud_logging

sys.path.insert(0, '/app')
sys.path.insert(0, '/app/ModelDevelopment/RAG')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MedScan RAG")

STARTUP_ERROR = None
READY = False

# Pre-loaded resources (initialized at startup)
_rag_config = None
_embeddings_data = None
_faiss_index = None
_llm_model = None
_embedding_model = None

def normalize_config(config: dict) -> dict:
    """
    Normalize MLflow-generated config to match ModelFactory expectations.
    Handles various config formats without modifying the source file.
    
    Args:
        config: Raw config from MLflow or other source
        
    Returns:
        Normalized config ready for RAG pipeline
    """
    normalized = config.copy()
    
    # 1. Fix model_name: Convert HuggingFace format to ModelFactory key
    model_name = normalized.get('model_name', '')
    
    # Map HuggingFace model names to ModelFactory keys
    model_mapping = {
        'Qwen/Qwen2.5-1.5B-Instruct': 'qwen_2.5_1.5b',
        'Qwen/Qwen2.5-7B-Instruct': 'qwen_2.5_7b',
        'Qwen/Qwen2.5-14B-Instruct': 'qwen_2.5_14b',
        'meta-llama/Llama-3.2-3B-Instruct': 'llama_3.2_3b',
        'meta-llama/Meta-Llama-3.1-8B-Instruct': 'llama_3.1_8b',
        'mistralai/Mistral-7B-Instruct-v0.3': 'mistral_7b',
        'HuggingFaceTB/SmolLM2-360M': 'smol_lm',
    }
    
    if model_name in model_mapping:
        logger.info(f"Normalizing model_name: {model_name} -> {model_mapping[model_name]}")
        normalized['model_name'] = model_mapping[model_name]
    elif model_name not in ['qwen_2.5_1.5b', 'qwen_2.5_7b', 'qwen_2.5_14b', 
                             'llama_3.2_3b', 'llama_3.1_8b', 'mistral_7b', 'smol_lm']:
        logger.warning(f"Unknown model_name: {model_name}, using qwen_2.5_1.5b as default")
        normalized['model_name'] = 'qwen_2.5_1.5b'
    
    # 2. Ensure model_type is set
    if 'model_type' not in normalized or normalized.get('model_type') == 'open-source':
        normalized['model_type'] = normalized['model_name']
        logger.info(f"Set model_type to: {normalized['model_type']}")
    
    # 3. Add max_tokens if missing (required by generate_response)
    if 'max_tokens' not in normalized:
        normalized['max_tokens'] = 500
        logger.info(f"Added missing max_tokens: {normalized['max_tokens']}")
    
    # 4. Validate and adjust temperature (0.0 to 1.0)
    temperature = normalized.get('temperature', 0.7)
    if temperature > 1.0:
        logger.warning(f"Temperature {temperature} > 1.0, capping at 1.0")
        normalized['temperature'] = 1.0
    elif temperature < 0.0:
        logger.warning(f"Temperature {temperature} < 0.0, setting to 0.1")
        normalized['temperature'] = 0.1
    
    # 5. Validate and adjust top_p (0.0 to 1.0)
    top_p = normalized.get('top_p', 0.9)
    if top_p > 1.0:
        logger.warning(f"top_p {top_p} > 1.0, capping at 1.0")
        normalized['top_p'] = 1.0
    elif top_p < 0.0:
        logger.warning(f"top_p {top_p} < 0.0, setting to 0.1")
        normalized['top_p'] = 0.1
    
    # 6. Validate k (number of documents to retrieve)
    k = normalized.get('k', 5)
    if k < 1:
        logger.warning(f"k={k} < 1, setting to 3")
        normalized['k'] = 3
    elif k > 10:
        logger.warning(f"k={k} > 10, capping at 10")
        normalized['k'] = 10
    
    # 7. Validate retrieval_method
    valid_methods = ['similarity', 'weighted_score']
    retrieval_method = normalized.get('retrieval_method', 'similarity')
    if retrieval_method not in valid_methods:
        logger.warning(f"Unknown retrieval_method: {retrieval_method}, using 'similarity'")
        normalized['retrieval_method'] = 'similarity'
    
    # 8. Ensure embedding_model is set
    if 'embedding_model' not in normalized:
        normalized['embedding_model'] = 'BAAI/llm-embedder'
        logger.info(f"Added default embedding_model: {normalized['embedding_model']}")
    
    # 9. Fix truncated or missing prompt
    prompt = normalized.get('prompt', '')
    if not prompt or len(prompt) < 50 or not '{context}' in prompt or not '{query}' in prompt:
        logger.warning("Prompt is missing, truncated, or malformed. Using default prompt.")
        normalized['prompt'] = """You are a medical information assistant that provides evidence-based responses using only the provided medical literature.

CONTEXT DOCUMENTS:
{context}

QUESTION:
{query}

INSTRUCTIONS:
1. Answer the question based ONLY on the information in the context documents above
2. Cite specific details from the documents to support your answer
3. If the context doesn't contain enough information to fully answer the question, acknowledge this limitation
4. Provide a clear, concise, and medically accurate response
5. Use professional medical terminology while remaining understandable

ANSWER:"""
    
    logger.info("NORMALIZED CONFIG:")
    logger.info(f"  model_name: {normalized['model_name']}")
    logger.info(f"  model_type: {normalized['model_type']}")
    logger.info(f"  temperature: {normalized['temperature']}")
    logger.info(f"  top_p: {normalized['top_p']}")
    logger.info(f"  max_tokens: {normalized['max_tokens']}")
    logger.info(f"  k: {normalized['k']}")
    logger.info(f"  retrieval_method: {normalized['retrieval_method']}")
    logger.info(f"  embedding_model: {normalized['embedding_model']}")
    logger.info(f"  prompt_length: {len(normalized['prompt'])} chars")
    
    return normalized


@app.on_event("startup")
async def startup():
    global STARTUP_ERROR, READY, _rag_config, _embeddings_data, _faiss_index, _llm_model, _embedding_model
    try:
        # Set env vars
        bucket = os.getenv('GCS_BUCKET', 'medscan-pipeline-medscanai-476500')
        os.environ['GCS_BUCKET_NAME'] = bucket
        os.environ['GCS_BUCKET'] = bucket
        os.environ['GCP_PROJECT_ID'] = 'medscanai-476500'
        
        logger.info(f"Environment set: bucket={bucket}")
        
        logger.info("DEVICE INFORMATION")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU count: {torch.cuda.device_count()}")
        else:
            logger.info("Running on CPU only")
        
        from ModelInference.RAG_inference import load_config
        
        # Load raw config
        logger.info("Loading RAG configuration...")
        raw_config = load_config()
        logger.info(f"Raw config loaded: model_name={raw_config.get('model_name')}")
        
        # Normalize config to handle MLflow format
        normalized_config = normalize_config(raw_config)
        _rag_config = normalized_config
        
        # Store normalized config globally for backward compatibility
        import ModelInference.RAG_inference as rag_module
        rag_module._NORMALIZED_CONFIG = normalized_config
        logger.info("Normalized config stored globally")
        
        # Load data files
        from ModelInference.RAG_inference import load_embeddings_data, load_faiss_index
        
        logger.info("LOADING DATA FILES")
        _embeddings_data = load_embeddings_data()
        _faiss_index = load_faiss_index()
        
        if _embeddings_data is None:
            raise RuntimeError("Failed to load embeddings data")
        if _faiss_index is None:
            raise RuntimeError("Failed to load FAISS index")
        
        logger.info(f"Embeddings: {len(_embeddings_data)} records")
        logger.info(f"FAISS index: {_faiss_index.ntotal} vectors, dimension={_faiss_index.d}")
        
        # Pre-load embedding model
        logger.info("Loading embedding model...")
        from sentence_transformers import SentenceTransformer
        embedding_model_name = _rag_config.get('embedding_model', 'BAAI/llm-embedder')
        _embedding_model = SentenceTransformer(
            embedding_model_name, 
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        logger.info(f"Embedding model loaded: {embedding_model_name}")
        
        # Pre-load LLM model
        logger.info("Loading LLM model...")
        from models.models import ModelFactory
        model_name = _rag_config.get('model_name')
        temperature = _rag_config.get('temperature', 0.7)
        top_p = _rag_config.get('top_p', 0.9)
        max_tokens = _rag_config.get('max_tokens', 500)
        
        _llm_model = ModelFactory.create_model(
            model_name,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )
        logger.info(f"LLM model loaded: {model_name} (temperature={temperature}, top_p={top_p}, max_tokens={max_tokens})")
        
        READY = True
        logger.info("SERVICE READY ✓ (All models pre-loaded)")
    except Exception as e:
        STARTUP_ERROR = str(e)
        logger.error(f"STARTUP FAILED: {e}")
        import traceback
        traceback.print_exc()


@app.get("/health")
def health():
    return {
        "status": "healthy" if READY else "failed",
        "ready": READY,
        "error": STARTUP_ERROR
    }


@app.get("/config")
def get_config():
    """Get current normalized configuration"""
    if not READY:
        return {"error": "Service not ready"}
    
    try:
        if _rag_config:
            return {
                "model_name": _rag_config.get('model_name'),
                "model_type": _rag_config.get('model_type'),
                "temperature": _rag_config.get('temperature'),
                "top_p": _rag_config.get('top_p'),
                "max_tokens": _rag_config.get('max_tokens'),
                "k": _rag_config.get('k'),
                "retrieval_method": _rag_config.get('retrieval_method'),
                "embedding_model": _rag_config.get('embedding_model'),
            }
        return {"error": "Config not found"}
    except Exception as e:
        return {"error": str(e)}


def _get_embedding(query: str) -> Optional[np.ndarray]:
    """
    Generate embedding using pre-loaded embedding model.
    
    Args:
        query: User query string
        
    Returns:
        Embedding vector or None on failure
    """
    global _embedding_model
    try:
        if not query or not query.strip():
            logger.error("Empty query provided")
            return None
        
        embedding = _embedding_model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        if embedding is None or len(embedding) == 0:
            logger.error("Generated embedding is empty")
            return None
        
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        import traceback
        traceback.print_exc()
        return None


def _generate_response_with_preloaded_model(
    query: str,
    documents: List[Dict[str, Any]]
) -> Optional[Tuple[str, int, int]]:
    """
    Generate response using pre-loaded LLM model.
    
    Args:
        query: User query
        documents: Retrieved documents with metadata
        
    Returns:
        Tuple of (response string, input_tokens, output_tokens) or None on failure
    """
    global _llm_model, _rag_config
    try:
        prompt_template = _rag_config.get("prompt", "")
        
        if not prompt_template:
            raise ValueError("Prompt template not found in config")
        
        # Format context from documents with TRUNCATION for memory safety
        context_parts = []
        for d in documents:
            title = d.get('title', 'Unknown')
            content = d.get('content', '')
            
            # CRITICAL: Truncate long content to prevent OOM
            if len(content) > 2000:  # Max 2000 chars per document
                content = content[:2000] + "...[truncated]"
            
            context_parts.append(f"Document {d['rank']} - {title}:\n{content}")
        
        context = "\n\n".join(context_parts)
        
        # CRITICAL: Truncate total context if too long (for L4 GPU)
        max_context_length = 6000  # Conservative limit
        if len(context) > max_context_length:
            context = context[:max_context_length] + "\n\n[Context truncated due to length...]"
            logger.warning(f"Context truncated to {max_context_length} characters")
        
        # Format prompt
        formatted_prompt = prompt_template.format(context=context, query=query)
        
        # Generate response using pre-loaded model
        response_d = _llm_model.infer(formatted_prompt)
        
        if not response_d or response_d.get('success') != True:
            raise ValueError(f"Model inference failed: {response_d}")
        
        response = response_d.get('generated_text', '')
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
        
        # Add medical disclaimer
        footer = (
            "\n\n---\n"
            "**Important:** This information is for educational purposes only and "
            "should not replace professional medical advice. Please consult a "
            "healthcare provider for diagnosis, treatment, or medical guidance."
        )
        response += footer
        
        logger.info(f"Response generated successfully: {in_tokens} input tokens, {out_tokens} output tokens")
        return response, in_tokens, out_tokens
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        import traceback
        traceback.print_exc()
        return None


def _create_lightweight_stats(
    query: str,
    response: str,
    retrieved_docs: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Create lightweight stats with query, response, and sources.
    This is returned immediately to the user.
    
    Args:
        query: User query
        response: Generated response
        retrieved_docs: Retrieved documents
        
    Returns:
        Dictionary with query, response, sources, and basic metrics
    """
    sources = []
    scores = []
    
    for doc in retrieved_docs:
        source_info = {
            "rank": doc.get("rank"),
            "title": doc.get("title", "Unknown"),
            "chunk_id": doc.get("chunk_id"),
            "score": round(doc.get("score", 0.0), 4),
            "link": doc.get("metadata", {}).get("link", "")
        }
        sources.append(source_info)
        
        # Collect scores for avg_retrieval_score calculation
        # Same calculation as compute_stats() for consistency
        score = doc.get("score", 0.0)
        scores.append(score)
    
    # Calculate average retrieval score (same as compute_stats)
    # These are REAL scores from FAISS distance: score = 1 / (1 + distance)
    avg_retrieval_score = round(sum(scores) / len(scores), 4) if scores else 0.0
    
    return {
        "query": query,
        "response": response,
        "sources": sources,
        "num_sources": len(sources),
        "num_retrieved_docs": len(retrieved_docs),  # For client compatibility
        "avg_retrieval_score": avg_retrieval_score  # For client compatibility (confidence)
    }


@app.post("/predict")
async def predict(body: dict, background_tasks: BackgroundTasks):
    """
    Process RAG query and return response immediately.
    Uses pre-loaded models for efficiency.
    Full metrics computation happens in background to reduce latency.
    """
    global _rag_config, _embeddings_data, _faiss_index
    
    if not READY:
        return {
            "predictions": [{
                "error": STARTUP_ERROR or "Service not ready",
                "success": False
            }]
        }
    
    try:
        # Extract query from instances
        instances = body.get("instances", [])
        if not instances:
            return {
                "predictions": [{
                    "error": "No instances provided in request body",
                    "success": False
                }]
            }
        
        query = instances[0].get("query")
        if not query:
            return {
                "predictions": [{
                    "error": "No query provided in instance",
                    "success": False
                }]
            }
        
        logger.info(f"Processing query: {query[:100]}...")
        
        # Track execution time (critical path - user waits for this)
        start_time = time.time()
        
        # Step 1: Generate embedding using pre-loaded model
        embedding = _get_embedding(query)
        if embedding is None:
            error_msg = "Failed to generate query embedding."
            logger.error(error_msg)
            return {
                "predictions": [{
                    "error": error_msg,
                    "success": False
                }]
            }
        
        # Step 2: Retrieve documents using pre-loaded index
        from ModelInference.RAG_inference import retrieve_documents
        k = _rag_config.get("k", 5)
        retrieval_method = _rag_config.get("retrieval_method", "similarity")
        documents = retrieve_documents(embedding, _faiss_index, _embeddings_data, k, retrieval_method)
        
        if not documents:
            error_msg = "Failed to retrieve documents."
            logger.error(error_msg)
            return {
                "predictions": [{
                    "error": error_msg,
                    "success": False
                }]
            }
        
        logger.info(f"Retrieved {len(documents)} documents")
        
        # Step 3: Generate response using pre-loaded LLM model
        result = _generate_response_with_preloaded_model(query, documents)
        if result is None:
            error_msg = "Failed to generate response from LLM."
            logger.error(error_msg)
            return {
                "predictions": [{
                    "error": error_msg,
                    "success": False
                }]
            }
        
        response, in_tokens, out_tokens = result
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Step 4: Create lightweight stats for immediate return
        lightweight_stats = _create_lightweight_stats(query, response, documents)
        
        # Step 5: Queue full stats computation and logging as background task
        # This happens AFTER response is sent to user
        background_tasks.add_task(
            _compute_and_log_full_stats,
            query=query,
            response=response,
            documents=documents,
            in_tokens=in_tokens,
            out_tokens=out_tokens,
            response_time=response_time
        )
        
        # Return lightweight stats immediately - user doesn't wait for full stats/logging
        logger.info(f"Query processed successfully ({in_tokens + out_tokens} tokens, {response_time:.3f}s)")
        return {
            "predictions": [{
                "answer": response,
                "stats": lightweight_stats,
                "success": True
            }]
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to get query for error logging
        error_query = None
        try:
            error_query = query  # query variable from try block
        except NameError:
            try:
                error_query = body.get("instances", [{}])[0].get("query") if body else None
            except:
                pass
        
        # Queue error logging as background task
        background_tasks.add_task(
            _log_prediction_metrics,
            success=False,
            error=str(e),
            response_time=0.0,
            query=error_query
        )
        
        return {
            "predictions": [{
                "error": str(e),
                "success": False
            }]
        }


def _compute_and_log_full_stats(
    query: str,
    response: str,
    documents: List[Dict[str, Any]],
    in_tokens: int,
    out_tokens: int,
    response_time: float
):
    """
    Compute full statistics and log to Cloud Logging.
    This runs in background after response is sent to user.
    
    Args:
        query: User query
        response: Generated response
        documents: Retrieved documents with metadata
        in_tokens: Number of input tokens
        out_tokens: Number of output tokens
        response_time: Total execution time
    """
    global _rag_config, _faiss_index
    try:
        # Compute full stats using the compute_stats function from RAG_inference
        from ModelInference.RAG_inference import compute_stats
        
        prompt_template = _rag_config.get("prompt", "")
        stats = compute_stats(
            query,
            response,
            documents,
            _rag_config,
            prompt_template,
            in_tokens,
            out_tokens,
            _faiss_index.ntotal if _faiss_index else 0
        )
        
        # Calculate composite score using same heuristic as model selection
        # Composite score = semantic_score * 0.5 + hallucination_score * 0.5
        # In production, we use avg_retrieval_score as proxy for semantic_score
        # (both measure semantic relevance of retrieved documents to query)
        hallucination_score = stats.get('hallucination_scores', {}).get('avg', 0.0) if stats else 0.0
        avg_retrieval_score = stats.get('avg_retrieval_score', 0.0) if stats else 0.0
        # Use retrieval_score as semantic_score proxy to match model selection heuristic
        composite_score = avg_retrieval_score * 0.5 + hallucination_score * 0.5
        
        # Extract document indices (doc_id from retrieved documents) for embedding space usage tracking
        retrieved_doc_indices = stats.get('retrieved_doc_indices', []) if stats else []
        
        # Log metrics to Cloud Logging
        _log_prediction_metrics(
            success=True,
            response_time=response_time,
            query=query,
            composite_score=composite_score,
            hallucination_score=hallucination_score,
            avg_retrieval_score=avg_retrieval_score,
            retrieved_doc_indices=retrieved_doc_indices,
            stats=stats
        )
        
        logger.info(f"Background metrics logged: composite_score={composite_score:.4f}, latency={response_time:.3f}s")
    except Exception as e:
        # Don't fail if background task has errors
        logger.warning(f"Failed to compute and log full stats in background: {e}")
        import traceback
        traceback.print_exc()


def _log_prediction_metrics(
    success: bool,
    response_time: float,
    query: str = None,
    composite_score: float = None,
    hallucination_score: float = None,
    avg_retrieval_score: float = None,
    retrieved_doc_indices: list = None,
    stats: dict = None,
    error: str = None
):
    """
    Log prediction metrics to Cloud Logging for monitoring.
    Does NOT store query or response text for privacy.
    
    Args:
        success: Whether prediction succeeded
        response_time: Execution time in seconds
        composite_score: Calculated composite score
        hallucination_score: Hallucination score from stats
        avg_retrieval_score: Average retrieval score
        retrieved_doc_indices: List of document indices (doc_id) retrieved
        stats: Full stats dictionary from RAG pipeline
        error: Error message if failed
    """
    try:
        project_id = os.getenv('GCP_PROJECT_ID', 'medscanai-476500')
        logger.info(f"Initializing Cloud Logging client for project: {project_id}")
        logging_client = cloud_logging.Client(project=project_id)
        logger_instance = logging_client.logger("rag_predictions")
        logger.info(f"Cloud Logging logger instance created: rag_predictions")
        
        # Build metrics payload (NO query/response text)
        payload = {
            "prediction_result": {
                "success": success,
                "latency": response_time,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        }
        
        if query is not None:
            payload["prediction_result"]["query"] = query
        
        # Add composite score and component metrics if available
        if composite_score is not None:
            payload["prediction_result"]["composite_score"] = round(composite_score, 4)
        
        if hallucination_score is not None:
            payload["prediction_result"]["hallucination_score"] = round(hallucination_score, 4)
        
        if avg_retrieval_score is not None:
            payload["prediction_result"]["avg_retrieval_score"] = round(avg_retrieval_score, 4)
        
        # Add retrieved document indices (for embedding space usage tracking)
        if retrieved_doc_indices:
            payload["prediction_result"]["retrieved_doc_indices"] = retrieved_doc_indices
            payload["prediction_result"]["num_retrieved_docs"] = len(retrieved_doc_indices)
        
        # Add existing inference metrics from stats
        if stats:
            if 'input_tokens' in stats:
                payload["prediction_result"]["input_tokens"] = stats.get('input_tokens', 0)
            if 'output_tokens' in stats:
                payload["prediction_result"]["output_tokens"] = stats.get('output_tokens', 0)
            if 'total_tokens' in stats:
                payload["prediction_result"]["total_tokens"] = stats.get('total_tokens', 0)
            if 'num_retrieved_docs' in stats:
                payload["prediction_result"]["num_retrieved_docs"] = stats.get('num_retrieved_docs', 0)
            # Store retrieval metrics per document (without content) - for detailed analysis
            if 'retrieved_docs_metrics' in stats:
                payload["prediction_result"]["retrieved_docs_metrics"] = stats.get('retrieved_docs_metrics', [])
            # NEW: Store index_size for embedding space coverage calculation
            if 'index_size' in stats:
                payload["prediction_result"]["index_size"] = stats.get('index_size', 0)
        
        # Add error if failed
        if error:
            payload["prediction_result"]["error"] = error
        
        # Log to Cloud Logging
        try:
            logger.info(f"Attempting to log to Cloud Logging with payload keys: {list(payload.keys())}")
            logger_instance.log_struct(payload, severity="INFO")
            logger.info(f"✅ log_struct() call completed successfully")
        except Exception as log_error:
            logger.error(f"❌ log_struct() failed: {log_error}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise  # Re-raise to be caught by outer exception handler
        
        if success:
            logger.info(f"✅ Logged prediction metrics to Cloud Logging: success={success}, composite_score={composite_score if composite_score else 'N/A'}, latency={response_time:.3f}s")
        else:
            logger.error(f"✅ Logged prediction metrics to Cloud Logging: success={success}, error={error}, latency={response_time:.3f}s")
    except Exception as e:
        # Don't fail the request if logging fails, but log the error with full details
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"❌ Failed to log prediction metrics to Cloud Logging: {e}")
        logger.error(f"Error details: {error_details}")
        logger.error(f"Payload that failed to log: {json.dumps(payload, default=str)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))