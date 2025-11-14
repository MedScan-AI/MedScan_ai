import numpy as np
import faiss
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def retrieve_documents(
    embedding: np.ndarray,
    index: faiss.Index,
    embeddings_data: List[Dict[str, Any]],
    k: int,
    retrieval_method: str = "similarity"
) -> Optional[List[Dict[str, Any]]]:
    """
    Retrieve top-k documents from FAISS index using specified method.
    
    Args:
        embedding: Query embedding vector
        index: FAISS index containing document embeddings
        embeddings_data: List of embedding records with metadata
        k: Number of documents to retrieve
        retrieval_method: Method for retrieval
            - "similarity": Basic cosine similarity (default)
            - "mmr": Maximal Marginal Relevance (diversity)
            - "weighted_score": Combined similarity + metadata weighting
            - "rerank": Retrieve more, then rerank top-k
        
    Returns:
        List of retrieved documents with full content and metadata or None on failure
    """
    try:
        logger.info(f"Retrieving {k} documents using {retrieval_method} method")
        
        # Reshape embedding for FAISS (needs 2D array)
        query_vector = embedding.reshape(1, -1).astype('float32')
        
        # Route to appropriate retrieval method
        if retrieval_method == "similarity":
            documents = _retrieve_similarity(query_vector, index, embeddings_data, k)
        
        elif retrieval_method == "mmr":
            documents = _retrieve_mmr(query_vector, index, embeddings_data, k, lambda_param=0.7)
        
        elif retrieval_method == "weighted_score":
            documents = _retrieve_weighted(query_vector, index, embeddings_data, k)
        
        elif retrieval_method == "rerank":
            documents = _retrieve_rerank(query_vector, index, embeddings_data, k)
        
        else:
            logger.warning(f"Unknown retrieval method '{retrieval_method}', using similarity")
            documents = _retrieve_similarity(query_vector, index, embeddings_data, k)
        
        logger.info(f"Retrieved {len(documents)} documents")
        return documents
        
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        return None


def _retrieve_similarity(
    query_vector: np.ndarray,
    index: faiss.Index,
    embeddings_data: List[Dict[str, Any]],
    k: int
) -> List[Dict[str, Any]]:
    """
    Basic similarity search - retrieve top-k by cosine similarity.
    """
    # Search FAISS index
    distances, indices = index.search(query_vector, k)
    
    # Format results
    documents = []
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx != -1 and idx < len(embeddings_data):
            record = embeddings_data[idx]
            documents.append({
                "rank": i + 1,
                "doc_id": idx,
                "chunk_id": record.get("chunk_id"),
                "title": record.get("title"),
                "content": record.get("content"),
                "metadata": record.get("metadata", {}),
                "distance": float(dist),
                "score": float(1 / (1 + dist)),
                "method": "similarity"
            })
    
    return documents


def _retrieve_mmr(
    query_vector: np.ndarray,
    index: faiss.Index,
    embeddings_data: List[Dict[str, Any]],
    k: int,
    lambda_param: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Maximal Marginal Relevance (MMR) retrieval.
    Balances relevance with diversity to avoid redundant results.
    
    MMR = λ * Similarity(query, doc) - (1-λ) * max(Similarity(doc, selected_docs))
    
    Args:
        lambda_param: Trade-off between relevance (1.0) and diversity (0.0)
                     Default 0.7 means 70% relevance, 30% diversity
    """
    # Retrieve more candidates than needed (2-3x)
    candidate_k = min(k * 3, len(embeddings_data))
    distances, indices = index.search(query_vector, candidate_k)
    
    # Get candidate embeddings
    candidate_indices = indices[0]
    valid_candidates = [idx for idx in candidate_indices if idx != -1 and idx < len(embeddings_data)]
    
    if not valid_candidates:
        return []
    
    # Get embeddings for candidates
    candidate_embeddings = index.reconstruct_batch(valid_candidates)
    
    # Calculate relevance scores (similarity to query)
    relevance_scores = 1 / (1 + distances[0][:len(valid_candidates)])
    
    # MMR selection
    selected_indices = []
    selected_embeddings = []
    
    for _ in range(min(k, len(valid_candidates))):
        if not selected_indices:
            # First document: highest relevance
            best_idx = 0
        else:
            # Calculate MMR scores for remaining candidates
            mmr_scores = []
            for i, candidate_idx in enumerate(valid_candidates):
                if candidate_idx in selected_indices:
                    mmr_scores.append(-float('inf'))
                    continue
                
                # Relevance component
                relevance = relevance_scores[i]
                
                # Diversity component: max similarity to already selected docs
                candidate_emb = candidate_embeddings[i:i+1]
                selected_emb = np.array(selected_embeddings)
                
                # Calculate cosine similarity with selected docs
                similarities = np.dot(selected_emb, candidate_emb.T).flatten()
                max_similarity = np.max(similarities) if len(similarities) > 0 else 0
                
                # MMR formula
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
                mmr_scores.append(mmr_score)
            
            best_idx = np.argmax(mmr_scores)
        
        selected_indices.append(valid_candidates[best_idx])
        selected_embeddings.append(candidate_embeddings[best_idx])
    
    # Format results
    documents = []
    for rank, idx in enumerate(selected_indices):
        record = embeddings_data[idx]
        # Find original distance
        original_pos = valid_candidates.index(idx)
        dist = distances[0][original_pos]
        
        documents.append({
            "rank": rank + 1,
            "doc_id": idx,
            "chunk_id": record.get("chunk_id"),
            "title": record.get("title"),
            "content": record.get("content"),
            "metadata": record.get("metadata", {}),
            "distance": float(dist),
            "score": float(1 / (1 + dist)),
            "method": "mmr"
        })
    
    return documents


def _retrieve_weighted(
    query_vector: np.ndarray,
    index: faiss.Index,
    embeddings_data: List[Dict[str, Any]],
    k: int
) -> List[Dict[str, Any]]:
    """
    Weighted retrieval combining similarity with metadata-based scoring.
    Useful if you have document quality scores, recency, etc.
    """
    # Retrieve more candidates to rerank
    candidate_k = min(k * 2, len(embeddings_data))
    distances, indices = index.search(query_vector, candidate_k)
    
    # Score each candidate
    scored_docs = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1 or idx >= len(embeddings_data):
            continue
        
        record = embeddings_data[idx]
        
        # Base similarity score
        similarity_score = 1 / (1 + dist)
        
        # Extract metadata weights (customize based on your metadata)
        metadata = record.get("metadata", {})
        
        # Example: boost recent documents
        recency_weight = metadata.get("recency_score", 1.0)  # 1.0 if not available
        
        # Example: boost high-quality sources
        quality_weight = metadata.get("quality_score", 1.0)
        
        # Example: boost documents with more citations
        citation_weight = min(1.0 + metadata.get("citation_count", 0) * 0.1, 2.0)
        
        # Combined weighted score
        # Adjust weights based on your use case
        final_score = (
            similarity_score * 0.7 +
            recency_weight * 0.1 +
            quality_weight * 0.1 +
            (citation_weight - 1.0) * 0.1  # Normalized to 0-0.1 range
        )
        
        scored_docs.append({
            "idx": idx,
            "record": record,
            "distance": dist,
            "similarity_score": similarity_score,
            "final_score": final_score
        })
    
    # Sort by final score and take top-k
    scored_docs.sort(key=lambda x: x["final_score"], reverse=True)
    top_docs = scored_docs[:k]
    
    # Format results
    documents = []
    for rank, doc in enumerate(top_docs):
        documents.append({
            "rank": rank + 1,
            "doc_id": doc["idx"],
            "chunk_id": doc["record"].get("chunk_id"),
            "title": doc["record"].get("title"),
            "content": doc["record"].get("content"),
            "metadata": doc["record"].get("metadata", {}),
            "distance": float(doc["distance"]),
            "score": float(doc["final_score"]),
            "method": "weighted_score"
        })
    
    return documents


def _retrieve_rerank(
    query_vector: np.ndarray,
    index: faiss.Index,
    embeddings_data: List[Dict[str, Any]],
    k: int
) -> List[Dict[str, Any]]:
    """
    Two-stage retrieval: retrieve more candidates, then rerank using
    a more sophisticated scoring method.
    
    This is a simplified version - in production you might use:
    - Cross-encoder models
    - BM25 scoring
    - Custom reranking models
    """
    # Stage 1: Retrieve more candidates
    candidate_k = min(k * 3, len(embeddings_data))
    distances, indices = index.search(query_vector, candidate_k)
    
    # Stage 2: Rerank candidates
    reranked_docs = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1 or idx >= len(embeddings_data):
            continue
        
        record = embeddings_data[idx]
        content = record.get("content", "")
        
        # Base similarity
        base_score = 1 / (1 + dist)
        
        # Simple reranking heuristics (replace with actual reranker)
        # Example: penalize very short or very long content
        content_length = len(content.split())
        length_penalty = 1.0
        if content_length < 50:
            length_penalty = 0.8  # Too short
        elif content_length > 500:
            length_penalty = 0.9  # Too long
        
        # Example: boost if title is very relevant
        title = record.get("title", "").lower()
        # This is placeholder - in reality you'd do proper matching
        title_boost = 1.0
        
        # Combined rerank score
        rerank_score = base_score * length_penalty * title_boost
        
        reranked_docs.append({
            "idx": idx,
            "record": record,
            "distance": dist,
            "base_score": base_score,
            "rerank_score": rerank_score
        })
    
    # Sort by rerank score and take top-k
    reranked_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
    top_docs = reranked_docs[:k]
    
    # Format results
    documents = []
    for rank, doc in enumerate(top_docs):
        documents.append({
            "rank": rank + 1,
            "doc_id": doc["idx"],
            "chunk_id": doc["record"].get("chunk_id"),
            "title": doc["record"].get("title"),
            "content": doc["record"].get("content"),
            "metadata": doc["record"].get("metadata", {}),
            "distance": float(doc["distance"]),
            "score": float(doc["rerank_score"]),
            "method": "rerank"
        })
    
    return documents


# Example usage
if __name__ == "__main__":
    """
    Example of how different methods work:
    
    1. similarity: Fast, returns most similar docs
       Use when: Speed is critical, semantic similarity is primary concern
    
    2. mmr: Balances relevance and diversity
       Use when: You want diverse results, avoid redundancy
    
    3. weighted_score: Incorporates metadata signals
       Use when: You have quality/recency/authority metadata
    
    4. rerank: Two-stage retrieval for better precision
       Use when: Quality > speed, willing to trade latency for accuracy
    """
    pass