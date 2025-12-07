"""
RAG_serve.py - Enhanced version with robust config normalization
Handles MLflow-generated configs without modification
"""
import os
import sys
import logging
from fastapi import FastAPI
import uvicorn
import torch

sys.path.insert(0, '/app')
sys.path.insert(0, '/app/ModelDevelopment/RAG')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MedScan RAG")

STARTUP_ERROR = None
READY = False

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
    global STARTUP_ERROR, READY
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
        
        # Store normalized config globally for the pipeline to use
        import ModelInference.RAG_inference as rag_module
        rag_module._NORMALIZED_CONFIG = normalized_config
        logger.info("Normalized config stored globally")
        
        # Validate data files
        from ModelInference.RAG_inference import load_embeddings_data, load_faiss_index
        
        logger.info("VALIDATING DATA FILES")
        embeddings = load_embeddings_data()
        index = load_faiss_index()
        
        if embeddings is None:
            raise RuntimeError("Failed to load embeddings data")
        if index is None:
            raise RuntimeError("Failed to load FAISS index")
        
        logger.info(f"Embeddings: {len(embeddings)} records")
        logger.info(f"FAISS index: {index.ntotal} vectors, dimension={index.d}")
        
        READY = True
        logger.info("SERVICE READY ✓")
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
        import ModelInference.RAG_inference as rag_module
        config = getattr(rag_module, '_NORMALIZED_CONFIG', None)
        if config:
            return {
                "model_name": config.get('model_name'),
                "model_type": config.get('model_type'),
                "temperature": config.get('temperature'),
                "top_p": config.get('top_p'),
                "max_tokens": config.get('max_tokens'),
                "k": config.get('k'),
                "retrieval_method": config.get('retrieval_method'),
                "embedding_model": config.get('embedding_model'),
            }
        return {"error": "Config not found"}
    except Exception as e:
        return {"error": str(e)}


@app.post("/predict")
def predict(body: dict):
    if not READY:
        return {
            "predictions": [{
                "error": STARTUP_ERROR or "Service not ready",
                "success": False
            }]
        }
    
    try:
        # Import with normalized config
        import ModelInference.RAG_inference as rag_module
        
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
        
        # Run pipeline (will use normalized config)
        response, stats = rag_module.run_rag_pipeline(query)
        
        # Check if response indicates an error
        if response and (response.startswith("Error") or response.startswith("Failed")):
            logger.error(f"Pipeline returned error: {response}")
            return {
                "predictions": [{
                    "error": response,
                    "stats": stats,
                    "success": False
                }]
            }
        
        logger.info(f"✓ Query processed successfully ({stats.get('total_tokens', 0)} tokens)")
        return {
            "predictions": [{
                "answer": response,
                "stats": stats,
                "success": True
            }]
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "predictions": [{
                "error": str(e),
                "success": False
            }]
        }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))