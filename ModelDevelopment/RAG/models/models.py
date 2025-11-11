"""
models.py - Unified text generation classes with vLLM
Optimized for long-context summarization (15k-40k tokens)
"""

import logging
import torch
from typing import Dict, Any
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ------------------------------
# vLLM Models (Sorted by Size)
# ------------------------------

class Qwen25_7B_Instruct:
    """Qwen 2.5 7B Instruct via vLLM - 128K context"""

    def __init__(self, max_tokens=500, temperature=0.7, top_p=0.9, **kwargs):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

        logger.info("Loading Qwen/Qwen2.5-7B-Instruct with vLLM")
        self.llm = LLM(
            model="Qwen/Qwen2.5-7B-Instruct",
            # tensor_parallel_size=1,
            dtype="bfloat16",
            enable_prefix_caching=False,  
            disable_log_stats=True,
            # max_model_len=32768,  # Adjust based on your k value
        )
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
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
            logger.error(f"Qwen 2.5-7B inference error: {e}")
            return {"generated_text": f"Error: {e}", "success": False}

class Mistral7BInstruct:
    """Mistral 7B Instruct v0.3 via vLLM - 32K context"""

    def __init__(self, max_tokens=500, temperature=0.7, top_p=0.9, **kwargs):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

        logger.info("Loading mistralai/Mistral-7B-Instruct-v0.3 with vLLM")
        self.llm = LLM(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            # tensor_parallel_size=1,
            dtype="bfloat16",
            enable_prefix_caching=False,  
            disable_log_stats=True,
            # max_model_len=32768,
        )
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
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

class Llama31_8B_Instruct:
    """Meta Llama 3.1 8B Instruct via vLLM - 128K context"""

    def __init__(self, max_tokens=500, temperature=0.7, top_p=0.9, **kwargs):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

        logger.info("Loading meta-llama/Meta-Llama-3.1-8B-Instruct with vLLM")
        self.llm = LLM(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            # tensor_parallel_size=1,
            dtype="bfloat16",
            enable_prefix_caching=False, 
            disable_log_stats=True,
            # max_model_len=32768,
        )
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
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
            logger.error(f"Llama 3.1 inference error: {e}")
            return {"generated_text": f"Error: {e}", "success": False}

class Qwen25_14B_Instruct:
    """Qwen 2.5 14B Instruct via vLLM - 128K context (BEST for summarization)"""

    def __init__(self, max_tokens=500, temperature=0.7, top_p=0.9, **kwargs):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

        logger.info("Loading Qwen/Qwen2.5-14B-Instruct with vLLM")
        self.llm = LLM(
            model="Qwen/Qwen2.5-14B-Instruct",
            # tensor_parallel_size=1,
            dtype="bfloat16",
            enable_prefix_caching=False,  
            disable_log_stats=True,
            # max_model_len=32768,
        )
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
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
            logger.error(f"Qwen 2.5-14B inference error: {e}")
            return {"generated_text": f"Error: {e}", "success": False}

class SmolLM2:
    def __init__(self, max_tokens=500, temperature=0.7, top_p=0.9, **kwargs):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        model_id = "HuggingFaceTB/SmolLM2-360M"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    
    def infer(self, prompt: str, **kwargs) -> Dict[str, Any]:
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_length = inputs["input_ids"].shape[1]
            outputs = self.model.generate(**inputs, max_new_tokens=50)
            generated_tokens = outputs[0][input_length:]
            output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True) 
            return {
                "generated_text": output_text,
                "input_tokens": int(input_length),
                "output_tokens": len(generated_tokens),
                "success": True,
            }
        except Exception as e:
            logger.error(f"SmolLM2 inference error: {e}")
            return {"generated_text": f"Error: {e}", "success": False}

class Qwen25_1_5B_Instruct:
    """Qwen 2.5 1.5B Instruct via vLLM - 128K context (BEST for efficiency & multilingual)"""

    def __init__(self, max_tokens=500, temperature=0.7, top_p=0.9, **kwargs):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

        logger.info("Loading Qwen/Qwen2.5-1.5B-Instruct with vLLM")
        self.llm = LLM(
            model="Qwen/Qwen2.5-1.5B-Instruct",
            dtype="bfloat16",
            enable_prefix_caching=True,
            disable_log_stats=True,
            # max_model_len=100000,  # Uncomment to limit to 100K context
        )
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
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
            logger.error(f"Qwen 2.5-1.5B inference error: {e}")
            return {"generated_text": f"Error: {e}", "success": False}

class Llama32_3B_Instruct:
    """Llama 3.2 3B Instruct via vLLM - 128K context (BEST overall performance)"""

    def __init__(self, max_tokens=500, temperature=0.7, top_p=0.9, **kwargs):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

        logger.info("Loading meta-llama/Llama-3.2-3B-Instruct with vLLM")
        self.llm = LLM(
            model="meta-llama/Llama-3.2-3B-Instruct",
            dtype="bfloat16",
            enable_prefix_caching=True,
            disable_log_stats=True,
            # max_model_len=100000,  # Uncomment to limit to 100K context
        )
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
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
            logger.error(f"Llama 3.2-3B inference error: {e}")
            return {"generated_text": f"Error: {e}", "success": False}

# ------------------------------
# Model Factory
# ------------------------------

class ModelFactory:
    """Factory for unified model creation"""

    MODEL_CLASSES = {
        "qwen_2.5_7b": Qwen25_7B_Instruct,
        "mistral_7b": Mistral7BInstruct,
        "llama_3.1_8b": Llama31_8B_Instruct,
        "qwen_2.5_14b": Qwen25_14B_Instruct,
        "smol_lm" : SmolLM2,
        "llama_3.2_3b": Llama32_3B_Instruct,
        "qwen_2.5_1.5b": Qwen25_1_5B_Instruct,
    }

    @staticmethod
    def create_model(
        model_key: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 500,
        **kwargs,
    ):
        if model_key not in ModelFactory.MODEL_CLASSES:
            raise ValueError(
                f"Unknown model '{model_key}'. Available: {list(ModelFactory.MODEL_CLASSES.keys())}"
            )
        cls = ModelFactory.MODEL_CLASSES[model_key]
        return cls(temperature=temperature, top_p=top_p, max_tokens=max_tokens, **kwargs)

    @staticmethod
    def list_models() -> Dict[str, str]:
        return {
            "qwen_2.5_7b": "Qwen 2.5 7B Instruct (7B, 128K ctx, vLLM)",
            "mistral_7b": "Mistral 7B Instruct v0.3 (7B, 32K ctx, vLLM)",
            "llama_3.1_8b": "Meta Llama 3.1 8B Instruct (8B, 128K ctx, vLLM)",
            "qwen_2.5_14b": "Qwen 2.5 14B Instruct (14B, 128K ctx, vLLM)",

            "smol_lm": "SmolLM2 (360M)",
            "llama_3.2_3b": "Llama (3B)",
            "qwen_2.5_1.5b": "Qwen Instruct multilingual (1.5B)"
        }