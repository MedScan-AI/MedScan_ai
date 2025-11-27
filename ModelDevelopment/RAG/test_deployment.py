"""
test_deployment.py - Test RAG deployment components locally
Run this before triggering Cloud Build
"""
import os
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_environment():
    """Check environment variables"""
    logger.info("Checking environment variables")
    
    required_vars = ['GCP_PROJECT_ID', 'GCS_BUCKET_NAME']
    missing = []
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            logger.info(f"  {var}: {value}")
        else:
            missing.append(var)
            logger.error(f"  {var}: NOT SET")
    
    if missing:
        logger.error(f"Missing environment variables: {missing}")
        logger.info("Set them in airflow/.env or export them")
        return False
    
    return True


def check_gcs_data():
    """Verify RAG data exists in GCS"""
    from google.cloud import storage
    
    logger.info("Checking GCS data")
    
    bucket_name = os.getenv("GCS_BUCKET_NAME")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    required_files = [
        'RAG/index/index_latest.bin',
        'RAG/index/embeddings_latest.json'
    ]
    
    missing = []
    for file_path in required_files:
        blob = bucket.blob(file_path)
        if blob.exists():
            logger.info(f"  Found: {file_path}")
        else:
            logger.error(f"  Missing: {file_path}")
            missing.append(file_path)
    
    if missing:
        logger.error("Missing files in GCS. Run DataPipeline first:")
        logger.info("  docker-compose exec webserver airflow dags trigger rag_data_pipeline_dvc")
        return False
    
    return True


def test_data_loading():
    """Test loading data from GCS"""
    logger.info("Testing data loading")
    
    sys.path.insert(0, str(Path(__file__).parent / 'ModelInference'))
    from RAG_inference import load_embeddings_data, load_faiss_index
    
    embeddings = load_embeddings_data()
    if embeddings is None:
        logger.error("Failed to load embeddings")
        return False
    logger.info(f"  Loaded {len(embeddings)} embeddings")
    
    index = load_faiss_index()
    if index is None:
        logger.error("Failed to load FAISS index")
        return False
    logger.info(f"  Loaded FAISS index ({index.ntotal} vectors)")
    
    return True


def test_model_selection():
    """Test model selection locally"""
    logger.info("Testing model selection")
    
    sys.path.insert(0, str(Path(__file__).parent / 'ModelSelection'))
    
    qa_path = Path(__file__).parent / 'ModelSelection' / 'qa.json'
    if not qa_path.exists():
        logger.error(f"QA dataset not found: {qa_path}")
        return False
    
    logger.info(f"  Found QA dataset: {qa_path}")
    
    with open(qa_path, 'r') as f:
        import json
        qa_data = json.load(f)
        logger.info(f"  Loaded {len(qa_data)} QA pairs")
    
    return True


def test_config_generation():
    """Test config file generation"""
    logger.info("Testing config generation")
    
    config_path = Path(__file__).parent / 'utils' / 'RAG_config.json'
    
    if config_path.exists():
        logger.info(f"  Found config: {config_path}")
        
        with open(config_path, 'r') as f:
            import json
            config = json.load(f)
            
            required_keys = ['model_name', 'temperature', 'top_p', 'k', 'prompt']
            missing = [k for k in required_keys if k not in config]
            
            if missing:
                logger.error(f"  Missing keys in config: {missing}")
                return False
            
            logger.info(f"  Model: {config['model_name']}")
            logger.info(f"  Composite score: {config.get('performance_metrics', {}).get('composite_score', 'N/A')}")
    else:
        logger.warning(f"  Config not found: {config_path}")
        logger.info("  This is OK - will be generated during experiment")
    
    return True


def run_all_tests():
    """Run all tests"""
    logger.info("=" * 80)
    logger.info("RAG Deployment - Pre-Flight Checks")
    logger.info("=" * 80)
    
    tests = [
        ("Environment Variables", check_environment),
        ("GCS Data Availability", check_gcs_data),
        ("Data Loading", test_data_loading),
        ("Model Selection", test_model_selection),
        ("Config Generation", test_config_generation),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info("")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test '{test_name}' failed with exception: {e}")
            results.append((test_name, False))
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("Test Results")
    logger.info("=" * 80)
    
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        logger.info(f"  {test_name:30} {status}")
    
    all_passed = all(result for _, result in results)
    
    logger.info("=" * 80)
    if all_passed:
        logger.info("All tests passed - ready for deployment")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Trigger Cloud Build: gcloud builds submit --config=cloudbuild/rag-training.yaml")
        logger.info("  2. Or push to GitHub to auto-trigger")
        return True
    else:
        logger.error("Some tests failed - fix issues before deployment")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)