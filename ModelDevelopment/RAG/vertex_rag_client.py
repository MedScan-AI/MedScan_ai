"""
vertex_rag_client.py - Client for querying Vertex AI RAG endpoint
"""
import os
import logging
from typing import Dict, Any, Optional
from google.cloud import aiplatform
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VertexRAGClient:
    """Client for Vertex AI RAG endpoint"""
    
    def __init__(
        self,
        project_id: str = None,
        region: str = "us-central1",
        endpoint_name: str = "medscan-rag-endpoint"
    ):
        if project_id is None:
            project_id = os.getenv("GCP_PROJECT_ID")
            if not project_id:
                raise ValueError("GCP_PROJECT_ID not set")
        
        self.project_id = project_id
        self.region = region
        self.endpoint_name = endpoint_name
        
        aiplatform.init(project=project_id, location=region)
        
        self.endpoint = self._get_endpoint()
    
    def _get_endpoint(self) -> aiplatform.Endpoint:
        """Get Vertex AI endpoint"""
        endpoints = aiplatform.Endpoint.list(
            filter=f'display_name="{self.endpoint_name}"'
        )
        
        if not endpoints:
            raise ValueError(f"Endpoint not found: {self.endpoint_name}")
        
        endpoint = endpoints[0]
        logger.info(f"Connected to endpoint: {endpoint.resource_name}")
        return endpoint
    
    def query(
        self,
        question: str,
        temperature: float = None,
        top_p: float = None,
        max_tokens: int = 500
    ) -> Dict[str, Any]:
        """
        Query the RAG endpoint.
        
        Args:
            question: User question
            temperature: Sampling temperature
            top_p: Top-p sampling
            max_tokens: Max output tokens
        
        Returns:
            Response dictionary with answer and metadata
        """
        try:
            instances = [{
                "prompt": question,
                "temperature": temperature or 0.7,
                "top_p": top_p or 0.9,
                "max_tokens": max_tokens
            }]
            
            logger.info(f"Sending request to endpoint")
            prediction = self.endpoint.predict(instances=instances)
            
            response = prediction.predictions[0]
            
            result = {
                "answer": response.get("text", ""),
                "input_tokens": response.get("input_tokens", 0),
                "output_tokens": response.get("output_tokens", 0),
                "success": True
            }
            
            logger.info(f"Received response ({result['output_tokens']} tokens)")
            return result
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {
                "answer": f"Error: {str(e)}",
                "success": False
            }
    
    def batch_query(self, questions: list) -> list:
        """Process multiple questions"""
        results = []
        for q in questions:
            result = self.query(q)
            results.append(result)
        return results


def main():
    """Example usage"""
    client = VertexRAGClient()
    
    response = client.query("What are the symptoms of tuberculosis?")
    
    if response['success']:
        print(response['answer'])
        print(f"\nTokens: {response['input_tokens']} in, {response['output_tokens']} out")
    else:
        print(f"Error: {response['answer']}")


if __name__ == "__main__":
    main()