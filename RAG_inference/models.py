"""
models.py - Optimized model classes for medical RAG inference
Uses the best free models that handle citations and long contexts well
All models work with prompts.py templates
"""

import logging
import torch
from typing import Dict, Any, Optional
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)
from prompts import get_prompt_template


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FlanT5Large:
    """Google Flan-T5 Large"""
    
    def __init__(self, max_tokens=500, temperature=0.7, top_p=0.9, **kwargs):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        logger.info("Loading Flan-T5 Large")
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
        
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
            logger.info("✓ Flan-T5 Base loaded on GPU")
        else:
            logger.info("✓ Flan-T5 Base loaded on CPU")
    
    def infer(
        self,
        query: str,
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate with T5 Base - good for simple Q&A"""
        try:
            # Build prompt
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                top_p=kwargs.get("top_p", self.top_p),
                do_sample=True,
                num_beams=2,
            )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return {
                "generated_text": generated_text,
                "input_tokens": len(inputs['input_ids'][0]),
                "output_tokens": len(outputs[0]),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            return {
                "generated_text": f"Error: {str(e)}",
                "input_tokens": 0,
                "output_tokens": 0,
                "success": False
            }


class ModelFactory:
    """Factory for creating model instances"""
    
    MODEL_CLASSES = {
        # Smaller/faster options
        "flan_t5": FlanT5Large,
    }
    
    @staticmethod
    def create_model(
        model_key: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ):
        """
        Create model instance
        
        Args:
            model_key: One of the available model keys
            max_tokens: Maximum generation length
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            top_p: Nucleus sampling (0.9 = focused, 0.95 = diverse)
            
        Returns:
            Model instance with infer() method
        """
        if model_key not in ModelFactory.MODEL_CLASSES:
            available = ", ".join(ModelFactory.MODEL_CLASSES.keys())
            raise ValueError(f"Unknown model: {model_key}. Available: {available}")
                
        model_class = ModelFactory.MODEL_CLASSES[model_key]
        return model_class(

            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
    
    @staticmethod
    def list_models() -> Dict[str, str]:
        """List all available models with recommendations"""
        return {
            "flan_t5": "Flan-T5 Base 250M - Lightweight, runs on CPU, simple Q&A",
        }