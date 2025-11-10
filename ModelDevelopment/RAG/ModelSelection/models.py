"""
models.py - Unified text generation classes with vLLM + Transformers
Supports both decoder-only (vLLM) and encoder-decoder (HF) models
"""

import logging
import torch
from typing import Dict, Any
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ------------------------------
# vLLM Models
# ------------------------------

class Llama31Instruct:
    """Meta Llama 3.1 8B Instruct via vLLM"""

    def __init__(self, max_tokens=500, temperature=0.7, top_p=0.9, **kwargs):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

        logger.info("Loading meta-llama/Llama-3.1-8B-Instruct with vLLM")
        self.llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", tensor_parallel_size=1)
        self.sampling_params = SamplingParams(
            temperature=temperature, top_p=top_p, max_tokens=max_tokens
        )

    def infer(self, prompt: str, **kwargs) -> Dict[str, Any]:
        try:
            outputs = self.llm.generate([prompt], sampling_params=self.sampling_params)
            out = outputs[0].outputs[0]
            return {
                "generated_text": out.text,
                "input_tokens": len(outputs[0].prompt_token_ids),
                "output_tokens": len(out.token_ids),
                "success": True,
            }
        except Exception as e:
            logger.error(f"Llama inference error: {e}")
            return {"generated_text": f"Error: {e}", "success": False}

class Mistral7BInstruct:
    """Mistral 7B Instruct via vLLM"""

    def __init__(self, max_tokens=500, temperature=0.7, top_p=0.9, **kwargs):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

        logger.info("Loading mistralai/Mistral-7B-Instruct-v0.2 with vLLM")
        self.llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2", tensor_parallel_size=1)
        self.sampling_params = SamplingParams(
            temperature=temperature, top_p=top_p, max_tokens=max_tokens
        )

    def infer(self, prompt: str, **kwargs) -> Dict[str, Any]:
        try:
            outputs = self.llm.generate([prompt], sampling_params=self.sampling_params)
            out = outputs[0].outputs[0]
            return {
                "generated_text": out.text,
                "input_tokens": len(outputs[0].prompt_token_ids),
                "output_tokens": len(out.token_ids),
                "success": True,
            }
        except Exception as e:
            logger.error(f"Mistral inference error: {e}")
            return {"generated_text": f"Error: {e}", "success": False}

# ------------------------------
# Transformers Models
# ------------------------------

class FlanT5Base:
    """Google Flan-T5 Base (encoder-decoder)"""

    def __init__(self, max_tokens=500, temperature=0.7, top_p=0.9, **kwargs):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

        logger.info("Loading google/flan-t5-base (Transformers)")
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")

    def infer(self, prompt: str, **kwargs) -> Dict[str, Any]:
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                top_p=kwargs.get("top_p", self.top_p),
                do_sample=True,
            )
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return {
                "generated_text": text,
                "input_tokens": len(inputs["input_ids"][0]),
                "output_tokens": len(outputs[0]),
                "success": True,
            }
        except Exception as e:
            logger.error(f"Flan-T5 inference error: {e}")
            return {"generated_text": f"Error: {e}", "success": False}

class Bloom560M:
    """BigScience BLOOM-560M (Transformers fallback)"""

    def __init__(self, max_tokens=500, temperature=0.7, top_p=0.9, **kwargs):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

        logger.info("Loading bigscience/bloom-560m (Transformers)")
        self.tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
        self.model = AutoModelForCausalLM.from_pretrained(
            "bigscience/bloom-560m", trust_remote_code=True
        )
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")

    def infer(self, prompt: str, **kwargs) -> Dict[str, Any]:
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                top_p=kwargs.get("top_p", self.top_p),
                do_sample=True,
            )
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return {
                "generated_text": text,
                "input_tokens": len(inputs["input_ids"][0]),
                "output_tokens": len(outputs[0]),
                "success": True,
            }
        except Exception as e:
            logger.error(f"BLOOM inference error: {e}")
            return {"generated_text": f"Error: {e}", "success": False}

# ------------------------------
# Model Factory
# ------------------------------

class ModelFactory:
    """Factory for unified model creation"""

    MODEL_CLASSES = {
        "llama_3_1": Llama31Instruct,
        "mistral_7b": Mistral7BInstruct,
        "flan_t5": FlanT5Base,
        "bloom_560m": Bloom560M,
    }

    @staticmethod
    def create_model(
        model_key: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ):
        if model_key not in ModelFactory.MODEL_CLASSES:
            raise ValueError(
                f"Unknown model '{model_key}'. Available: {list(ModelFactory.MODEL_CLASSES.keys())}"
            )
        cls = ModelFactory.MODEL_CLASSES[model_key]
        return cls(max_tokens=max_tokens, temperature=temperature, top_p=top_p, **kwargs)

    @staticmethod
    def list_models() -> Dict[str, str]:
        return {
            "llama_3_1": "Meta Llama 3.1 8B Instruct (vLLM)",
            "mistral_7b": "Mistral 7B Instruct v0.2 (vLLM)",
            "flan_t5": "Google Flan-T5 Base (Transformers)",
            "bloom_560m": "BigScience BLOOM-560M (Transformers)",
        }
