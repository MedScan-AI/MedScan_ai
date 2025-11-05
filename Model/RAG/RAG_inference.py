# RAG_inference.py
import sys
import time
import torch
import numpy as np
import logging
import random
from typing import Dict, List, Any, Tuple, Optional
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import psutil
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global caches
_MODEL_CACHE: Dict[str, Any] = {}
_TOKENIZER_CACHE: Dict[str, Any] = {}
_EMBEDDING_MODEL = None
_EMBEDDING_TOKENIZER = None
_DOCUMENT_EMBEDDINGS_CACHE: Dict[int, np.ndarray] = {}  # keyed by index in document list

# --- Embedding model utilities ------------------------------------------------
def get_embedding_model(model_name: str = "BAAI/llm-embedder"):
    """Load and return the embedding model & tokenizer (cached)."""
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
    Compute dense embedding for `text`.
    Uses CLS token (first token) from last_hidden_state. Returns 1D numpy array.
    """
    model, tokenizer = get_embedding_model()
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    # Forward
    with torch.no_grad():
        outputs = model(**inputs)
        # use first token embedding (CLS) or pooled output if available
        if hasattr(outputs, "last_hidden_state"):
            emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        elif hasattr(outputs, "pooler_output"):
            emb = outputs.pooler_output.cpu().numpy()
        else:
            emb = outputs[0][:, 0, :].cpu().numpy()
    return emb.reshape(-1)

def precompute_document_embeddings(document_texts: List[str], force: bool = False) -> Dict[int, np.ndarray]:
    """
    Compute and cache embeddings for a list of documents.
    Returns a mapping from doc index -> embedding (1D numpy arrays).
    Set force=True to recompute regardless of cache.
    """
    global _DOCUMENT_EMBEDDINGS_CACHE
    if not force and len(_DOCUMENT_EMBEDDINGS_CACHE) == len(document_texts):
        logger.info("Document embeddings already precomputed (count=%d).", len(document_texts))
        return _DOCUMENT_EMBEDDINGS_CACHE

    logger.info("Precomputing %d document embeddings...", len(document_texts))
    _DOCUMENT_EMBEDDINGS_CACHE = {}
    for idx, doc in enumerate(document_texts):
        try:
            _DOCUMENT_EMBEDDINGS_CACHE[idx] = get_embeddings(doc)
        except Exception as e:
            logger.error("Failed to embed doc idx=%d: %s", idx, str(e))
            _DOCUMENT_EMBEDDINGS_CACHE[idx] = np.zeros(768, dtype=float)  # fallback
    return _DOCUMENT_EMBEDDINGS_CACHE

# --- Retrieval ----------------------------------------------------------------
def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def retrieve_documents(
    query: str,
    document_store: List[str],
    num_docs: int,
    retrieval_method: str = "hybrid",
    doc_embeddings: Optional[Dict[int, np.ndarray]] = None
) -> List[str]:
    """
    Retrieve top documents for query.
    - `doc_embeddings` should be a mapping idx->embedding produced by precompute_document_embeddings.
    """
    if not document_store:
        return []

    retrieval_method = retrieval_method.lower()
    query_terms = set(query.lower().split())

    if retrieval_method == "bm25":
        # Lightweight lexical scoring (toy BM25-like heuristic)
        scores = []
        for idx, doc in enumerate(document_store):
            doc_terms = set(doc.lower().split())
            overlap = len(query_terms.intersection(doc_terms))
            # length normalization to prefer shorter docs with same overlap
            score = overlap / (1.0 + math.log(1 + len(doc_terms)))
            scores.append((idx, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        top = [document_store[idx] for idx, _ in scores[:num_docs]]
        return top

    # For embedding & hybrid methods, require doc_embeddings (precomputed)
    if doc_embeddings is None or len(doc_embeddings) < len(document_store):
        # compute on the fly (works but slower)
        logger.debug("doc_embeddings missing or incomplete, computing on the fly for embedding retrieval (slower)")
        doc_embeddings = {i: get_embeddings(doc) for i, doc in enumerate(document_store)}

    # Query embedding
    query_emb = get_embeddings(query)

    similarities = []
    for idx, doc_emb in doc_embeddings.items():
        sim = _cosine_sim(query_emb, doc_emb)
        similarities.append((idx, sim))
    similarities.sort(key=lambda x: x[1], reverse=True)

    # If hybrid: combine with a small lexical score
    if retrieval_method == "hybrid":
        hybrid_scores = []
        for idx, sim in similarities:
            doc = document_store[idx]
            doc_terms = set(doc.lower().split())
            overlap = len(query_terms.intersection(doc_terms))
            lex_score = overlap / (1 + len(doc_terms))
            combined = 0.7 * sim + 0.3 * lex_score
            hybrid_scores.append((idx, combined))
        hybrid_scores.sort(key=lambda x: x[1], reverse=True)
        top = [document_store[idx] for idx, _ in hybrid_scores[:num_docs]]
        return top

    # embedding only
    top = [document_store[idx] for idx, _ in similarities[:num_docs]]
    return top

# --- Hallucination detection (placeholder) -----------------------------------
def batch_evaluate(pairs, model):
    """
    Evaluate multiple premise-hypothesis pairs at once.
    """
    try:
        with torch.no_grad():
            scores = model.predict(pairs)
            # Convert to Python list if it's a tensor
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().tolist()
            return scores
    except Exception as e:
        logger.error(f"Error during hallucination evaluation: {str(e)}")
        sys.exit(1)

def get_hallucination_score(generated_text: str, context_docs: List[str], aggregation = "mean") -> float:
    """
    Evaluate a model response against multiple retrieved documents.
    """

    model = AutoModelForSequenceClassification.from_pretrained(
            "vectara/hallucination_evaluation_model", 
            trust_remote_code=False
        )
    if not context_docs:
        logger.warning("No documents provided for hallucination evaluation")
        return {
            "score": 0.0,
            "is_hallucinated": True,
            "doc_scores": [],
            "aggregation": aggregation
        }
    
    # Create pairs of each document with the response
    pairs = [(doc, generated_text) for doc in context_docs]
    individual_scores = batch_evaluate(pairs, model)
    
    # Aggregate the scores
    if aggregation == "mean":
        final_score = sum(individual_scores) / len(individual_scores)
    elif aggregation == "min":
        final_score = min(individual_scores)
    elif aggregation == "max":
        final_score = max(individual_scores)
    else:
        logger.warning(f"Unknown aggregation method: {aggregation}, using mean")
        final_score = sum(individual_scores) / len(individual_scores)
    
    # Calculate hallucination threshold (typically 0.5 is used, but can be configured)
    threshold = 0.5
    is_hallucinated = final_score < threshold
    
    # return {
    #     "score": final_score,
    #     "is_hallucinated": is_hallucinated,
    #     "doc_scores": [
    #         {"doc_index": i, "doc": doc[:100] + "...", "score": score} 
    #         for i, (doc, score) in enumerate(zip(context_docs, individual_scores))
    #     ],
    #     "aggregation": aggregation
    # }
    return final_score

# --- Retrieval score ----------------------------------------------------------
def compute_retrieval_score(retrieved_docs: List[str], relevant_docs: List[str]) -> float:
    if not retrieved_docs or not relevant_docs:
        return 0.0
    retrieved_set = set(retrieved_docs)
    relevant_set = set(relevant_docs)
    correct = len(retrieved_set.intersection(relevant_set))
    return correct / len(retrieved_docs)

# --- Semantic similarity -----------------------------------------------------
def compute_semantic_similarity(generated_answer: str, reference_answer: str) -> float:
    gen_emb = get_embeddings(generated_answer)
    ref_emb = get_embeddings(reference_answer)
    return _cosine_sim(gen_emb, ref_emb)

# --- LLM generation helpers ---------------------------------------------------
def _simple_api_generate(prompt: str, model_name: str, temperature: float, top_p: float, max_tokens: int = 500) -> Tuple[str, int, int, bool]:
    """
    Placeholder for API model generation. Replace with actual generation using
    API keys in production. Ensure API keys are not present in code. Use Env variables
    Returns generated text and token estimates.
    """
    pass

def _local_generate_placeholder(prompt: str, model_name: str, temperature: float, top_p: float, max_tokens: int = 500) -> Tuple[str, int, int]:
    """
    Placeholder for local model generation. Replace with actual generation using
    AutoModelForCausalLM / llm frameworks (bitsandbytes, accelerate) in production.
    Returns generated text and token estimates.
    """
    pass

# --- Orchestrator ------------------------------------------------------------
def run_inference(
    query: str,
    document_store: List[str],
    reference_answer: Optional[str],
    relevant_docs: List[str],
    model_name: str,
    model_type: str,
    temperature: float,
    top_p: float,
    num_retrieved_docs: int,
    retrieval_method: str,
    prompt_type: str,
    prompt_template: str,
    model_config: Dict[str, Any],
    doc_embeddings: Optional[Dict[int, np.ndarray]] = None
) -> Dict[str, Any]:
    """
    Run full RAG flow: retrieve -> build prompt -> generate -> metrics.
    Returns dict with keys: metrics (dict), generated_text (str), retrieved_docs (list).
    """
    start_time = time.time()

    # 1) Retrieval
    retrieved_docs = retrieve_documents(
        query=query,
        document_store=document_store,
        num_docs=num_retrieved_docs,
        retrieval_method=retrieval_method,
        doc_embeddings=doc_embeddings
    )

    # 2) Prompt construction
    context = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(retrieved_docs)])
    prompt = prompt_template.replace("{query}", query).replace("{context}", context)

    # 3) Generation
    input_tokens = 0
    output_tokens = 0
    generated_text = ""
    api_success = False

    try:
        if model_type == "api":
            generated_text, input_tokens, output_tokens, api_success = _simple_api_generate(
                prompt=prompt,
                model_name=model_name,
                temperature=temperature,
                top_p=top_p,
                max_tokens=model_config.get("max_tokens", 500)
            )
        else:
            # local model placeholder - replace with real local inference
            # Make sure CUDA cache is in a safe state
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            generated_text, input_tokens, output_tokens = _local_generate_placeholder(
                prompt=prompt,
                model_name=model_name,
                temperature=temperature,
                top_p=top_p,
                max_tokens=model_config.get("max_tokens", 500)
            )
            api_success = True

    except Exception as e:
        logger.exception("Generation error: %s", str(e))
        generated_text = f"Error generating response: {str(e)}"
        input_tokens = len(prompt.split())
        output_tokens = len(generated_text.split())
        api_success = False

    # 4) Metrics computation
    metrics = {}
    end_time = time.time()
    runtime_ms = (end_time - start_time) * 1000.0

    # semantic_matching_score
    if reference_answer:
        try:
            metrics["semantic_matching_score"] = compute_semantic_similarity(generated_text, reference_answer)
        except Exception:
            metrics["semantic_matching_score"] = 0.0
    else:
        metrics["semantic_matching_score"] = 0.0

    # hallucination_score
    metrics["hallucination_score"] = get_hallucination_score(generated_text, retrieved_docs)

    # retrieval_score: need relevant_docs content vs retrieved_docs
    metrics["retrieval_score"] = compute_retrieval_score(retrieved_docs, relevant_docs)

    # token, runtime, cost, system metrics
    metrics["avg_input_tokens"] = input_tokens
    metrics["avg_output_tokens"] = output_tokens
    metrics["runtime_per_query_ms"] = runtime_ms

    # Cost estimation (very approximate)
    if model_type == "api":
        # if "gpt-4" in model_name.lower():
        #     in_cost = 0.03; out_cost = 0.06
        # elif "gpt-3.5" in model_name.lower():
        #     in_cost = 0.0015; out_cost = 0.002
        # else:
        #     in_cost = 0.01; out_cost = 0.03

        #######################
        in_cost = 100
        out_cost = 200 
        #######################
        metrics["cost_per_query_usd"] = (input_tokens / 1000.0 * in_cost) + (output_tokens / 1000.0 * out_cost)
        metrics["api_success_rate"] = 100.0 if api_success else 0.0
        metrics["memory_usage_mb"] = psutil.Process().memory_info().rss / (1024 * 1024)
        metrics["gpu_utilization_percent"] = 0.0  # real GPU metrics would need nvidia-smi/pynvml
    else:
        metrics["cost_per_query_usd"] = 0.0
        metrics["api_success_rate"] = 100.0 if api_success else 0.0
        try:
            metrics["memory_usage_mb"] = psutil.Process().memory_info().rss / (1024 * 1024)
        except Exception:
            metrics["memory_usage_mb"] = 0.0
        metrics["gpu_utilization_percent"] = 80.0 if torch.cuda.is_available() else 0.0

    return {"metrics": metrics, "generated_text": generated_text, "retrieved_docs": retrieved_docs}
