"""
RAG_inference.py - RAG inference using evaluate_rag.py infrastructure
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np

from config import ExperimentConfig
from evaluate_rag import RAGEvaluator, QAPair
from models import ModelFactory
from retrieval import DocumentRetriever, get_embeddings
from prompts import get_prompt_template


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def parse_qa_file(qa_file_path: Path) -> List[Dict[str, str]]:
    """
    Parse QA file with format: Q1: question? A: answer
    
    Args:
        qa_file_path: Path to QA file
        
    Returns:
        List of dicts with 'question_id', 'question', 'answer'
    """
    qa_pairs = []
    
    try:
        with open(qa_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by Q followed by number
        pattern = r'Q(\d+):\s*(.+?)\s*A:\s*(.+?)(?=Q\d+:|$)'
        matches = re.findall(pattern, content, re.DOTALL)
        
        for match in matches:
            q_num, question, answer = match
            qa_pairs.append({
                'question_id': f"Q{q_num}",
                'question': question.strip(),
                'answer': answer.strip()
            })
        
        logger.info(f"Parsed {len(qa_pairs)} QA pairs from {qa_file_path}")
        return qa_pairs
        
    except Exception as e:
        logger.error(f"Error parsing QA file: {str(e)}")
        return []



def evaluate_rag_system(
    qa_file_path: Path,
    config: ExperimentConfig,
    retriever: Optional[DocumentRetriever] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate RAG system on QA pairs from file using RAGEvaluator
    
    Args:
        qa_file_path: Path to QA file with format Q1: question? A: answer
        config: Experiment configuration
        retriever: Optional pre-loaded retriever (will create if None)
        verbose: Print detailed results
        
    Returns:
        Dictionary with:
            - individual_results: List of per-question results
            - mean_metrics: Average metrics across all questions
            - aggregated_metrics: Detailed statistics (mean, std, min, max)
            - config: Configuration used
            - num_questions: Number of questions evaluated
    """
    logger.info("=" * 80)
    logger.info("EVALUATING RAG SYSTEM WITH QA FILE")
    logger.info("=" * 80)
    
    # Parse QA file
    qa_pairs = parse_qa_file(qa_file_path)
    if not qa_pairs:
        logger.error("No QA pairs found in file")
        return {}
    
    logger.info(f"Loaded {len(qa_pairs)} QA pairs")
    
    # Update config to use the QA file path
    config.paths.qa_file = qa_file_path
    
    # Create evaluator
    evaluator = RAGEvaluator(config)
    
    # Load components (retriever, model, qa_pairs)
    if retriever is not None:
        evaluator.retriever = retriever
        evaluator.model = None  # Will be loaded
        evaluator.qa_pairs = []
    
    evaluator.load_components()
    
    # Override QA pairs with parsed ones
    evaluator.qa_pairs = [
        QAPair(
            question=qa['question'],
            reference_answer=qa['answer'],
            question_id=qa['question_id']
        )
        for qa in qa_pairs
    ]
    
    logger.info(f"Using model: {config.model.name}")
    logger.info(f"Using prompt: {config.prompt.prompt_id}")
    logger.info(f"Retrieving {config.retrieval.num_docs} documents per query")
    
    # Run evaluation
    results = evaluator.run_evaluation()
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total Questions: {results['num_queries']}")
    logger.info(f"\nMean Metrics:")
    
    agg_metrics = results['aggregated_metrics']
    logger.info(
        f"  Semantic Similarity: "
        f"{agg_metrics.get('avg_semantic_similarity', 0):.3f} "
        f"± {agg_metrics.get('std_semantic_similarity', 0):.3f}"
    )
    logger.info(
        f"  Keyword Precision: "
        f"{agg_metrics.get('avg_keyword_precision', 0):.3f} "
        f"± {agg_metrics.get('std_keyword_precision', 0):.3f}"
    )
    logger.info(
        f"  Keyword Recall: "
        f"{agg_metrics.get('avg_keyword_recall', 0):.3f} "
        f"± {agg_metrics.get('std_keyword_recall', 0):.3f}"
    )
    logger.info(
        f"  Keyword F1: "
        f"{agg_metrics.get('avg_keyword_f1', 0):.3f} "
        f"± {agg_metrics.get('std_keyword_f1', 0):.3f}"
    )
    logger.info(
        f"  Hallucination Score: "
        f"{agg_metrics.get('avg_hallucination_score', 0):.3f} "
        f"± {agg_metrics.get('std_hallucination_score', 0):.3f}"
    )
    logger.info(
        f"  Avg Input Tokens: "
        f"{agg_metrics.get('avg_input_tokens', 0):.0f}"
    )
    logger.info(
        f"  Avg Output Tokens: "
        f"{agg_metrics.get('avg_output_tokens', 0):.0f}"
    )
    logger.info(
        f"  Success Rate: "
        f"{agg_metrics.get('avg_success', 0)*100:.1f}%"
    )
    logger.info("=" * 80)
    
    # Format output for backward compatibility
    return {
        'individual_results': results['individual_results'],
        'mean_metrics': {
            f"mean_{k.replace('avg_', '')}": v 
            for k, v in agg_metrics.items() if k.startswith('avg_')
        },
        'aggregated_metrics': agg_metrics,
        'config': results['config'],
        'num_questions': results['num_queries']
    }



def query_rag_system(
    query: str,
    config: ExperimentConfig,
    retriever: Optional[DocumentRetriever] = None,
    model=None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Get RAG response for a single query
    
    Args:
        query: User question
        config: Experiment configuration
        retriever: Optional pre-loaded retriever (will create if None)
        model: Optional pre-loaded model (will create if None)
        verbose: Print detailed output
        
    Returns:
        Dictionary with:
            - query: Original question
            - answer: Generated answer
            - retrieved_docs: List of retrieved documents with metadata
            - metadata: Token counts, runtime, success flag, etc.
    """
    start_time = time.time()
    
    if verbose:
        logger.info("\n" + "=" * 80)
        logger.info("RAG QUERY")
        logger.info("=" * 80)
        logger.info(f"Query: {query}")
    
    # Create retriever if not provided
    if retriever is None:
        logger.info("Creating retriever...")
        retriever = DocumentRetriever(
            index_path=config.paths.index_file,
            chunks_path=config.paths.chunks_file,
            original_data_path=config.paths.original_data_file
        )
    
    # Create model if not provided
    if model is None:
        logger.info(f"Loading model: {config.model.name}")
        model = ModelFactory.create_model(
            model_key=config.model.name,
            max_tokens=config.model.max_tokens,
            temperature=config.model.temperature,
            top_p=config.model.top_p
        )
    
    # Get prompt template
    prompt_template = get_prompt_template(config.prompt.prompt_id)
    
    # Retrieve documents
    logger.info("Retrieving documents...")
    query_embedding = get_embeddings(query)
    documents = retriever.retrieve(
        query_embedding=query_embedding,
        num_docs=config.retrieval.num_docs,
        threshold=config.retrieval.similarity_threshold
    )
    
    if verbose:
        logger.info(f"Retrieved {len(documents)} documents:")
        for idx, doc in enumerate(documents, 1):
            logger.info(f"  {idx}. {doc.title}")
            logger.info(f"     {doc.get_metadata_str()}")
    
    # Format context
    context = retriever.format_context_for_prompt(documents)
    
    # Build full prompt
    full_prompt = prompt_template.replace(
        "{context}", context
    ).replace(
        "{query}", query
    )
    
    # Generate answer
    logger.info("Generating answer...")
    result = model.infer(
        query=query,
        temperature=config.model.temperature,
        prompt = full_prompt,
        top_p=config.model.top_p,
        max_tokens=config.model.max_tokens
    )
    
    runtime = (time.time() - start_time) * 1000
    
    if verbose:
        logger.info("\n" + "-" * 80)
        logger.info("ANSWER:")
        logger.info("-" * 80)
        logger.info(result['generated_text'])
        logger.info("-" * 80)
        logger.info(f"\nMetadata:")
        logger.info(f"  Model: {config.model.name}")
        logger.info(f"  Prompt: {config.prompt.prompt_id}")
        logger.info(f"  Input tokens: {result['input_tokens']}")
        logger.info(f"  Output tokens: {result['output_tokens']}")
        logger.info(f"  Runtime: {runtime:.0f}ms")
        logger.info(f"  Success: {result['success']}")
        logger.info("=" * 80)
    
    return {
        'query': query,
        'answer': result['generated_text'],
        'retrieved_docs': [
            {
                'title': doc.title,
                'metadata': doc.get_metadata_str(),
                'content': doc.content
            }
            for doc in documents
        ],
        'metadata': {
            'input_tokens': result['input_tokens'],
            'output_tokens': result['output_tokens'],
            'runtime_ms': runtime,
            'success': result['success'],
            'model': config.model.name,
            'prompt': config.prompt.prompt_id,
            'num_retrieved_docs': len(documents)
        }
    }


def quick_evaluate(
    qa_file: str,
    model: str = "flan_t5",
    num_docs: int = 5,
    prompt: str = "prompt1",
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Quick evaluation with minimal setup
    
    Args:
        qa_file: Path to QA file
        model: Model key (flan_t5, phi3_mini, etc.)
        num_docs: Number of documents to retrieve
        prompt: Prompt template ID
        output_file: Optional path to save results JSON
        
    Returns:
        Evaluation results dictionary
    """
    config = ExperimentConfig()
    config.model.name = model
    config.retrieval.num_docs = num_docs
    config.prompt.prompt_id = prompt
    
    results = evaluate_rag_system(
        qa_file_path=Path(qa_file),
        config=config,
        verbose=True
    )
    
    if output_file:
        logger.info(f"Saving results to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
    
    return results


def quick_query(
    query: str,
    model: str = "flan_t5",
    num_docs: int = 5,
    prompt: str = "prompt1"
) -> str:
    """
    Quick query with minimal setup
    
    Args:
        query: Question to ask
        model: Model key
        num_docs: Number of documents to retrieve
        prompt: Prompt template ID
        
    Returns:
        Generated answer string
    """
    config = ExperimentConfig()
    config.model.name = model
    config.retrieval.num_docs = num_docs
    config.prompt.prompt_id = prompt
    
    response = query_rag_system(
        query=query,
        config=config,
        verbose=True
    )
    
    return response['answer']