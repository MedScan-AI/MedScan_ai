"""
RAG_serve.py - FastAPI server for serving RAG model on Vertex AI
Uses your actual code structure from ModelDevelopment/RAG/
"""
import os
import sys
import json
import logging
import time
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from google.cloud import storage, logging as cloud_logging
import numpy as np

sys.path.insert(0, '/app')
sys.path.insert(0, '/app/rag')

from rag.models.models import ModelFactory
from rag.ModelInference.RAG_inference import (
    load_embeddings_data,
    load_faiss_index,
    get_embedding
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MedScan RAG Service")

# Global state
MODEL = None
INDEX = None
EMBEDDINGS_DATA = None
CONFIG = None

# Cloud Logging setup
try:
    logging_client = cloud_logging.Client()
    cloud_logger = logging_client.logger('rag-predictions')
    logger.info("Cloud Logging initialized")
except Exception as e:
    logger.warning(f"Cloud Logging not available: {e}")
    cloud_logger = None


class PredictionRequest(BaseModel):
    """Request schema for predictions"""
    query: str
    k: int = 5
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None


class RetrievedDoc(BaseModel):
    """Schema for retrieved document"""
    text: str
    source: str
    score: float
    rank: int


class PredictionResponse(BaseModel):
    """Response schema for predictions"""
    answer: str
    retrieved_docs: List[RetrievedDoc]
    model_used: str
    tokens: Dict[str, int]
    success: bool


def download_from_gcs(bucket_name: str, source_path: str, dest_path: str):
    """Download file from GCS"""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_path)
        
        # Create parent directory if needed
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        blob.download_to_filename(dest_path)
        logger.info(f"✅ Downloaded {source_path}")
    except Exception as e:
        logger.error(f"❌ Failed to download {source_path}: {e}")
        raise


def load_config_from_gcs(bucket_name: str) -> Dict:
    """Load model configuration from GCS"""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # Try to get latest deployment info
        try:
            blob = bucket.blob("RAG/deployments/latest.txt")
            build_id = blob.download_as_text().strip()
            logger.info(f"Latest build ID: {build_id}")
            
            # Get config for this deployment
            config_blob = bucket.blob(f"RAG/deployments/{build_id}/model_config.json")
            config = json.loads(config_blob.download_as_text())
            logger.info(f"Loaded config from deployment {build_id}")
            return config
        except Exception as e:
            logger.warning(f"Could not load deployment config: {e}")
            
            # Fallback: Use default config
            return {
                "model_name": "Qwen/Qwen2.5-7B-Instruct",
                "display_name": "qwen_2.5_7b",
                "temperature": 0.7,
                "top_p": 0.9,
                "k": 5,
                "embedding_model": "BAAI/llm-embedder"
            }
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Initialize model and data on startup"""
    global MODEL, INDEX, EMBEDDINGS_DATA, CONFIG
    
    bucket_name = os.getenv("GCS_BUCKET")
    logger.info(f"🚀 Starting RAG service...")
    logger.info(f"Bucket: {bucket_name}")
    
    try:
        # Step 1: Load configuration
        logger.info("Step 1: Loading configuration...")
        CONFIG = load_config_from_gcs(bucket_name)
        logger.info(f"Config: {json.dumps(CONFIG, indent=2)}")
        
        # Step 2: Download FAISS index and embeddings
        logger.info("Step 2: Downloading FAISS index and embeddings...")
        download_from_gcs(
            bucket_name,
            "RAG/index/index_latest.bin",
            "/tmp/index/index.bin"
        )
        download_from_gcs(
            bucket_name,
            "RAG/index/embeddings_latest.json",
            "/tmp/index/embeddings.json"
        )
        
        # Step 3: Set environment variables for your inference code
        os.environ['FAISS_INDEX_PATH'] = '/tmp/index/index.bin'
        os.environ['EMBEDDINGS_PATH'] = '/tmp/index/embeddings.json'
        os.environ['GCS_BUCKET_NAME'] = bucket_name
        
        # Step 4: Load embeddings and FAISS index using your functions
        logger.info("Step 3: Loading embeddings and FAISS index...")
        EMBEDDINGS_DATA = load_embeddings_data()
        INDEX = load_faiss_index()
        
        if EMBEDDINGS_DATA is None or INDEX is None:
            raise Exception("Failed to load embeddings or FAISS index")
        
        logger.info(f"✅ Loaded {len(EMBEDDINGS_DATA)} embeddings")
        logger.info(f"✅ FAISS index has {INDEX.ntotal} vectors")
        
        # Step 5: Initialize model using your ModelFactory
        logger.info("Step 4: Initializing model...")
        model_key = CONFIG.get('display_name', 'qwen_2.5_7b')
        
        MODEL = ModelFactory.create_model(
            model_key=model_key,
            temperature=CONFIG.get('temperature', 0.7),
            top_p=CONFIG.get('top_p', 0.9),
            max_tokens=CONFIG.get('max_tokens', 500)
        )
        
        logger.info(f"✅ Model initialized: {model_key}")
        logger.info("="*60)
        logger.info("🎉 RAG SERVICE READY!")
        logger.info(f"Model: {model_key}")
        logger.info(f"Embeddings: {len(EMBEDDINGS_DATA)}")
        logger.info(f"FAISS vectors: {INDEX.ntotal}")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize service: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint with service info"""
    return {
        "service": "MedScan RAG Service",
        "status": "running",
        "model": CONFIG.get("display_name") if CONFIG else "unknown",
        "endpoints": {
            "health": "/health",
            "predict": "/predict"
        },
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if MODEL is None or INDEX is None or EMBEDDINGS_DATA is None:
        raise HTTPException(
            status_code=503,
            detail="Service not ready - model, index, or embeddings not loaded"
        )
    
    return {
        "status": "healthy",
        "model": CONFIG.get("display_name") if CONFIG else "unknown",
        "embeddings_count": len(EMBEDDINGS_DATA),
        "faiss_vectors": INDEX.ntotal
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Main prediction endpoint
    
    Request:
        {
            "query": "What are the symptoms of tuberculosis?",
            "k": 5,
            "temperature": 0.7,
            "max_tokens": 500
        }
    
    Response:
        {
            "answer": "Generated answer with references...",
            "retrieved_docs": [...],
            "model_used": "qwen_2.5_7b",
            "tokens": {"input": 123, "output": 456},
            "success": true
        }
    """
    start_time = time.time()
    
    try:
        # Validate service is ready
        if MODEL is None or INDEX is None or EMBEDDINGS_DATA is None:
            raise HTTPException(503, "Service not initialized")
        
        logger.info(f"📝 Query: {request.query[:100]}...")
        
        # Step 1: Get query embedding
        embedding_model = CONFIG.get('embedding_model', 'BAAI/llm-embedder')
        query_embedding = get_embedding(request.query, embedding_model)
        
        if query_embedding is None:
            raise HTTPException(500, "Failed to generate query embedding")
        
        # Step 2: Search FAISS index
        k = request.k or CONFIG.get('k', 5)
        query_vector = np.array(query_embedding).reshape(1, -1).astype('float32')
        
        distances, indices = INDEX.search(query_vector, k)
        
        # Step 3: Get retrieved documents
        retrieved_docs = []
        for rank, (idx, score) in enumerate(zip(indices[0], distances[0]), 1):
            if idx < len(EMBEDDINGS_DATA):
                doc = EMBEDDINGS_DATA[idx]
                retrieved_docs.append({
                    "text": doc.get("text", ""),
                    "source": doc.get("source", doc.get("metadata", {}).get("link", "Unknown")),
                    "title": doc.get("title", "Unknown"),
                    "score": float(score),
                    "rank": rank,
                    "metadata": doc.get("metadata", {})
                })
        
        logger.info(f"🔍 Retrieved {len(retrieved_docs)} documents")
        
        # Step 4: Build context for generation
        context_parts = []
        for doc in retrieved_docs:
            title = doc.get('title', 'Unknown')
            content = doc.get('text', '')
            context_parts.append(
                f"Document {doc['rank']} - {title}:\n{content}"
            )
        
        context = "\n\n".join(context_parts)
        
        # Step 5: Create prompt
        prompt_template = CONFIG.get('prompt', """Based on the following medical documents, answer the question accurately and concisely.

Context:
{context}

Question: {query}

Answer:""")
        
        formatted_prompt = prompt_template.format(
            context=context,
            query=request.query
        )
        
        # Step 6: Generate response using your model
        temperature = request.temperature or CONFIG.get('temperature', 0.7)
        max_tokens = request.max_tokens or CONFIG.get('max_tokens', 500)
        top_p = request.top_p or CONFIG.get('top_p', 0.9)
        
        # Update model parameters if provided
        if hasattr(MODEL, 'sampling_params'):
            from vllm import SamplingParams
            MODEL.sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens
            )
        
        result = MODEL.infer(formatted_prompt)
        
        if not result or not result.get('success'):
            raise HTTPException(500, f"Model inference failed: {result}")
        
        answer = result.get('generated_text', '')
        input_tokens = result.get('input_tokens', 0)
        output_tokens = result.get('output_tokens', 0)
        
        # Step 7: Add references to answer
        references = "\n\n**References:**\n"
        for doc in retrieved_docs:
            source = doc.get('source', '')
            title = doc.get('title', 'Unknown')
            if source and source != 'Unknown':
                references += f"{doc['rank']}. [{title}]({source})\n"
            else:
                references += f"{doc['rank']}. {title}\n"
        
        answer += references
        
        # Calculate latency
        latency = time.time() - start_time
        
        # Log to Cloud Logging for monitoring
        if cloud_logger:
            try:
                cloud_logger.log_struct({
                    'prediction_result': True,
                    'query': request.query,
                    'latency': latency,
                    'retrieved_docs': [
                        {'score': d['score'], 'rank': d['rank']} 
                        for d in retrieved_docs
                    ],
                    'success': True,
                    'model': CONFIG.get('display_name'),
                    'tokens': {
                        'input': input_tokens,
                        'output': output_tokens
                    }
                })
            except Exception as e:
                logger.warning(f"Failed to log to Cloud Logging: {e}")
        
        logger.info(f"✅ Generated answer in {latency:.2f}s")
        logger.info(f"Tokens: {input_tokens} in, {output_tokens} out")
        
        # Return response
        return PredictionResponse(
            answer=answer,
            retrieved_docs=[
                RetrievedDoc(
                    text=d['text'][:300] + "..." if len(d['text']) > 300 else d['text'],
                    source=d['source'],
                    score=d['score'],
                    rank=d['rank']
                )
                for d in retrieved_docs
            ],
            model_used=CONFIG.get('display_name', 'unknown'),
            tokens={
                'input': input_tokens,
                'output': output_tokens,
                'total': input_tokens + output_tokens
            },
            success=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}", exc_info=True)
        
        # Log error to Cloud Logging
        if cloud_logger:
            try:
                cloud_logger.log_struct({
                    'prediction_result': True,
                    'query': request.query,
                    'success': False,
                    'error': str(e)
                })
            except:
                pass
        
        raise HTTPException(500, f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    logger.info(f"Starting server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)