"""
RAG_config.py - Configuration utilities for RAG deployment
"""
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from google.cloud import storage


def load_rag_config(
    config_path: str = "utils/RAG_config.json",
    bucket_name: str = None
) -> Dict[str, Any]:
    """
    Load RAG configuration from local or GCS.
    
    Args:
        config_path: Path to config (local or gs:// URI)
        bucket_name: GCS bucket name (from env if None)
    
    Returns:
        Configuration dictionary
    """
    if bucket_name is None:
        bucket_name = os.getenv("GCS_BUCKET_NAME", "medscan-pipeline-medscanai-476500")
    
    if config_path.startswith('gs://'):
        parts = config_path.replace('gs://', '').split('/', 1)
        bucket_name = parts[0]
        blob_path = parts[1]
        
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        
        content = blob.download_as_text()
        return json.loads(content)
    
    possible_paths = [
        Path(config_path),
        Path(__file__).parent / 'RAG_config.json',
        Path('/workspace/ModelDevelopment/RAG/utils/RAG_config.json'),
    ]
    
    for path in possible_paths:
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
    
    raise FileNotFoundError(f"Config not found: {config_path}")


def get_vllm_deployment_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract vLLM deployment configuration from RAG config.
    
    Args:
        config: RAG configuration dictionary
    
    Returns:
        vLLM deployment configuration
    """
    model_name = config.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct")
    
    model_size_map = {
        "1.5b": "small",
        "3b": "small",
        "7b": "small",
        "8b": "medium",
        "13b": "medium",
        "14b": "medium",
    }
    
    model_name_lower = model_name.lower()
    machine_size = "small"
    for size_key, machine in model_size_map.items():
        if size_key in model_name_lower:
            machine_size = machine
            break
    
    vllm_args = [
        "--host=0.0.0.0",
        "--port=7080",
        f"--model={model_name}",
        "--tensor-parallel-size=1",
        "--swap-space=16",
        "--gpu-memory-utilization=0.85",
        "--disable-log-stats",
        "--max-model-len=8192",
        "--trust-remote-code"
    ]
    
    return {
        "model_name": model_name,
        "vllm_args": vllm_args,
        "machine_size": machine_size,
        "temperature": config.get("temperature", 0.7),
        "top_p": config.get("top_p", 0.9),
        "max_tokens": 500
    }


def get_deployment_info(bucket_name: str = None) -> Optional[Dict[str, Any]]:
    """
    Get latest deployment info from GCS.
    
    Args:
        bucket_name: GCS bucket name
    
    Returns:
        Deployment info dictionary or None
    """
    if bucket_name is None:
        bucket_name = os.getenv("GCS_BUCKET_NAME", "medscan-pipeline-medscanai-476500")
    
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob("RAG/deployments/latest/deployment_info.json")
        
        if not blob.exists():
            return None
        
        content = blob.download_as_text()
        return json.loads(content)
        
    except Exception as e:
        print(f"Error loading deployment info: {e}")
        return None