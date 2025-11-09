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

class Bloom:
    """Fixed BLOOM model with repetition prevention"""
    
    def __init__(
        self, 
        model_name: str = "bigscience/bloom-560m",
        max_tokens: int = 500,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,  # ADD THIS
        **kwargs
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty  # ADD THIS
        
        logger.info(f"Loading BLOOM model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
            logger.info(f"✓ {model_name} loaded on GPU")
        else:
            logger.info(f"✓ {model_name} loaded on CPU")
    
    def infer(
        self,
        query: str,
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        try:

            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            input_length = inputs['input_ids'].shape[1]
            
            # FIXED GENERATION PARAMETERS
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                top_p=kwargs.get("top_p", self.top_p),
                top_k=50,  # ADD: Limit vocabulary
                repetition_penalty=kwargs.get("repetition_penalty", self.repetition_penalty),  # FIX
                no_repeat_ngram_size=3,  # ADD: Prevent 3-gram repetition
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                length_penalty=1.0,  # ADD: Encourage appropriate length
                bad_words_ids=None,  # Could add forbidden sequences
                min_length=10,  # Ensure some minimum response
            )
            
            # Decode
            generated_text = self.tokenizer.decode(
                outputs[0][input_length:],
                skip_special_tokens=True
            )
            
            # POST-PROCESSING: Remove repetitive patterns
            generated_text = self._remove_repetitions(generated_text)
            
            return {
                "generated_text": generated_text.strip(),
                "input_tokens": input_length,
                "output_tokens": outputs.shape[1] - input_length,
                "success": True
            }
        
        except Exception as e:
            logger.error(f"BLOOM generation error: {str(e)}")
            return {
                "generated_text": f"Error: {str(e)}",
                "input_tokens": 0,
                "output_tokens": 0,
                "success": False
            }
    
    def _remove_repetitions(self, text: str, max_repeat: int = 3) -> str:
        """Remove excessive repetitions from generated text"""
        import re
        
        # Find repeated phrases
        words = text.split()
        cleaned_words = []
        
        for i, word in enumerate(words):
            # Check if this starts a repetitive sequence
            if i < len(words) - max_repeat:
                phrase = " ".join(words[i:i+5])  # Check 5-word phrases
                remaining_text = " ".join(words[i+5:])
                if phrase in remaining_text:
                    # Found repetition, truncate here
                    return " ".join(cleaned_words)
            cleaned_words.append(word)
        
        return " ".join(cleaned_words)


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
        "bloom": Bloom
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