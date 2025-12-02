#!/usr/bin/env python
"""
run_inference.py - Run single queries or full evaluation on RAG system
Supports both interactive queries and batch evaluation
"""

import sys
import json
import logging
from pathlib import Path
from typing import Optional
import argparse

from inference import query_rag_system, evaluate_rag_system, quick_query
from config import ExperimentConfig
from retrieval import DocumentRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_single_query(
    query: str,
    index_path: Path,
    chunks_path: Path,
    original_data_path: Optional[Path] = None,
    model: str = "flan_t5",
    num_docs: int = 5,
    prompt_id: str = "prompt1",
    temperature: float = 0.7,
    save_output: Optional[Path] = None
) -> dict:
    """
    Run a single query against the RAG system
    
    Args:
        query: The question to ask
        index_path: Path to FAISS index.bin
        chunks_path: Path to chunks pickle file
        original_data_path: Path to original data JSON for metadata
        model: Model name to use
        num_docs: Number of documents to retrieve
        prompt_id: Prompt template to use
        temperature: Generation temperature
        save_output: Optional path to save the output
        
    Returns:
        Response dictionary
    """
    logger.info("=" * 80)
    logger.info("SINGLE QUERY MODE")
    logger.info("=" * 80)
    
    # Create configuration
    config = ExperimentConfig()
    config.model.name = model
    config.model.temperature = temperature
    config.retrieval.num_docs = num_docs
    config.prompt.prompt_id = prompt_id
    
    # Update paths
    config.paths.index_file = index_path
    config.paths.chunks_file = chunks_path
    if original_data_path:
        config.paths.original_data_file = original_data_path
    
    # Create retriever
    logger.info("Loading retriever...")
    retriever = DocumentRetriever(
        index_path=index_path,
        chunks_path=chunks_path,
        original_data_path=original_data_path
    )
    
    logger.info(f"Retriever stats: {retriever.get_stats()}")
    
    # Run query
    response = query_rag_system(
        query=query,
        config=config,
        retriever=retriever,
        verbose=True
    )
    
    # Save output if requested
    if save_output:
        logger.info(f"\nSaving response to: {save_output}")
        with open(save_output, 'w', encoding='utf-8') as f:
            json.dump(response, f, indent=2)
    
    return response


def run_evaluation(
    qa_file: Path = "data/qa.txt",
    index_path: Path = "data/index.bin",
    chunks_path: Path = "data/data.pkl",
    original_data_path: Optional[Path] = "data/scraped_updated.jsonl",
    model: str = "flan_t5",
    num_docs: int = 5,
    prompt_id: str = "prompt1",
    temperature: float = 0.7,
    save_output: Optional[Path] = None
) -> dict:
    """
    Run full evaluation with QA file
    
    Args:
        qa_file: Path to QA pairs file
        index_path: Path to FAISS index.bin
        chunks_path: Path to chunks pickle file
        original_data_path: Path to original data JSON
        model: Model name to use
        num_docs: Number of documents to retrieve
        prompt_id: Prompt template to use
        temperature: Generation temperature
        save_output: Optional path to save results
        
    Returns:
        Evaluation results dictionary
    """
    logger.info("=" * 80)
    logger.info("EVALUATION MODE")
    logger.info("=" * 80)
    
    # Create configuration
    config = ExperimentConfig()
    config.model.name = model
    config.model.temperature = temperature
    config.retrieval.num_docs = num_docs
    config.prompt.prompt_id = prompt_id
    
    # Update paths
    config.paths.index_file = index_path
    config.paths.chunks_file = chunks_path
    config.paths.qa_file = qa_file
    if original_data_path:
        config.paths.original_data_file = original_data_path
    
    # Run evaluation
    results = evaluate_rag_system(
        qa_file_path=qa_file,
        config=config,
        verbose=True
    )
    
    # Save results if requested
    if save_output:
        logger.info(f"\nSaving results to: {save_output}")
        with open(save_output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
    
    return results


def main():
    """Main entry point with CLI"""
    parser = argparse.ArgumentParser(
        description="Run RAG inference - single query or evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single query
  python run_inference.py query "What are the causes of lung cancer?" \\
      --index data/index.bin --chunks data/chunks.pkl

  # Evaluation with QA file
  python run_inference.py evaluate --qa-file data/qa_pairs.txt \\
      --index data/index.bin --chunks data/chunks.pkl
      
  # With custom settings
  python run_inference.py query "Your question here" \\
      --index data/index.bin --chunks data/chunks.pkl \\
      --model phi3_mini --num-docs 10 --temperature 0.5
        """
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Mode to run')
    
    # Query mode
    query_parser = subparsers.add_parser('query', help='Run single query')
    query_parser.add_argument(
        'question',
        type=str,
        help='Question to ask the RAG system'
    )
    query_parser.add_argument(
        '--index',
        type=Path,
        help='Path to FAISS index.bin file'
    )
    query_parser.add_argument(
        '--chunks',
        type=Path,
        help='Path to chunks pickle file'
    )
    query_parser.add_argument(
        '--original-data',
        type=Path,
        help='Path to original data JSON for metadata enrichment'
    )
    query_parser.add_argument(
        '--model',
        type=str,
        default='flan_t5',
        help='Model to use (default: flan_t5)'
    )
    query_parser.add_argument(
        '--num-docs',
        type=int,
        default=5,
        help='Number of documents to retrieve (default: 5)'
    )
    query_parser.add_argument(
        '--prompt',
        type=str,
        default='prompt1',
        help='Prompt template ID (default: prompt1)'
    )
    query_parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Generation temperature (default: 0.7)'
    )
    query_parser.add_argument(
        '--save-output',
        type=Path,
        help='Path to save the response JSON'
    )
    
    # Evaluation mode
    eval_parser = subparsers.add_parser('evaluate', help='Run evaluation with QA file')
    eval_parser.add_argument(
        '--qa-file',
        type=Path,
        required=True,
        help='Path to QA pairs file'
    )
    eval_parser.add_argument(
        '--index',
        type=Path,
        help='Path to FAISS index.bin file'
    )
    eval_parser.add_argument(
        '--chunks',
        type=Path,
        help='Path to chunks pickle file'
    )
    eval_parser.add_argument(
        '--original-data',
        type=Path,
        help='Path to original data JSON for metadata'
    )
    eval_parser.add_argument(
        '--model',
        type=str,
        default='flan_t5',
        help='Model to use (default: flan_t5)'
    )
    eval_parser.add_argument(
        '--num-docs',
        type=int,
        default=5,
        help='Number of documents to retrieve (default: 5)'
    )
    eval_parser.add_argument(
        '--prompt',
        type=str,
        default='prompt1',
        help='Prompt template ID (default: prompt1)'
    )
    eval_parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Generation temperature (default: 0.7)'
    )
    eval_parser.add_argument(
        '--save-output',
        type=Path,
        help='Path to save evaluation results JSON'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'query':
        # Run single query
        response = run_single_query(
            query=args.question,
            index_path=args.index,
            chunks_path=args.chunks,
            original_data_path=args.original_data,
            model=args.model,
            num_docs=args.num_docs,
            prompt_id=args.prompt,
            temperature=args.temperature,
            save_output=args.save_output
        )
        
        # Print just the answer for easy reading
        print("\n" + "="*80)
        print("ANSWER:")
        print("="*80)
        print(response['answer'])
        print("="*80)
        
    elif args.mode == 'evaluate':
        # Run evaluation
        results = run_evaluation(
            qa_file=args.qa_file,
            index_path=args.index,
            chunks_path=args.chunks,
            original_data_path=args.original_data,
            model=args.model,
            num_docs=args.num_docs,
            prompt_id=args.prompt,
            temperature=args.temperature,
            save_output=args.save_output
        )
        
        # Print summary
        agg = results['aggregated_metrics']
        print("\n" + "="*80)
        print("EVALUATION COMPLETE")
        print("="*80)
        print(f"Questions evaluated: {results['num_questions']}")
        print(f"Avg Semantic Similarity: {agg.get('avg_semantic_similarity', 0):.3f}")
        print(f"Avg Keyword F1: {agg.get('avg_keyword_f1', 0):.3f}")
        print(f"Avg Hallucination Score: {agg.get('avg_hallucination_score', 0):.3f}")
        print("="*80)
        
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)