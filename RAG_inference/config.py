"""
config.py - Configuration management for RAG evaluation system
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import yaml


@dataclass
class RetrievalConfig:
    """Configuration for document retrieval"""
    num_docs: int = 3
    method: str = "hybrid"  # Options: "bm25", "embedding", "hybrid"
    similarity_threshold: float = 0.0


@dataclass
class ModelConfig:
    """Configuration for LLM model"""
    name: str = "flan_t5"  # Model key or full name
    type: str = "huggingface"  # Kept for backwards compatibility
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 500


@dataclass
class PromptConfig:
    """Configuration for prompt selection"""
    prompt_id: str = "prompt1"  # Options: "prompt1" to "prompt5"


@dataclass
class PathConfig:
    """File paths for data and outputs"""
    embeddings_file: Path = Path("data/RAG/index/embeddings.json")
    index_file: Path = Path("data/RAG/index/index.bin")
    chunks_file: Path = Path("data/RAG/index/data.pkl")
    original_data_file: Path = Path("data/RAG/original_data/documents.json")
    qa_file: Path = Path("data/RAG/evaluation/qa_pairs.txt")
    mlflow_tracking_uri: Path = Path("mlruns")


@dataclass
class MLflowConfig:
    """MLflow experiment configuration"""
    experiment_name: str = "medical_rag_evaluation"
    run_name: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "retrieval": {
                "num_docs": self.retrieval.num_docs,
                "method": self.retrieval.method,
                "similarity_threshold": self.retrieval.similarity_threshold
            },
            "model": {
                "name": self.model.name,
                "type": self.model.type,
                "temperature": self.model.temperature,
                "top_p": self.model.top_p,
                "max_tokens": self.model.max_tokens,
            },
            "prompt": {
                "prompt_id": self.prompt.prompt_id
            }
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create configuration from dictionary"""
        retrieval = RetrievalConfig(**config_dict.get("retrieval", {}))
        model = ModelConfig(**config_dict.get("model", {}))
        prompt = PromptConfig(**config_dict.get("prompt", {}))
        paths = PathConfig(**config_dict.get("paths", {}))
        mlflow_cfg = MLflowConfig(**config_dict.get("mlflow", {}))
        
        return cls(
            retrieval=retrieval,
            model=model,
            prompt=prompt,
            paths=paths,
            mlflow=mlflow_cfg
        )
    
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> 'ExperimentConfig':
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def save_yaml(self, yaml_path: Path) -> None:
        """Save configuration to YAML file"""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


# Hyperparameter search space for grid search
HYPERPARAMETER_GRID = {
    "num_docs": [3, 5],
    # "retrieval_method": ["bm25", "embedding", "hybrid"],
    "retrieval_method": ["bm25"],
    "prompt_id": ["prompt1", "prompt2", "prompt3", "prompt4", "prompt5"],
    "temperature": [0.3, 0.7, 1.0],
    "model_name": [
       "flan_t5_base"
    ]
}

source_mappings = {
        "NIH/PubMed": [
            "nih.gov",
            "pubmed.ncbi.nlm.nih.gov",
            "ncbi.nlm.nih.gov/pmc",
            "pmc.ncbi.nlm.nih.gov"
        ],
        "Clinical Journal": [
            "jamanetwork.com",
            "ajronline.org",
            "ascopubs.org",
            "biomedcentral.com",
            "bmj.com",
            "thelancet.com",
            "nejm.org",
            "nature.com/articles",
            "sciencedirect.com",
            "mdpi.com",
            "jphe.amegroups.org",
            "ccts.amegroups.org",
            "e-emj.org"
        ],
        "Research Database": [
            "journals.plos.org",
            "plosone.org",
            "frontiersin.org",
            "hindawi.com",
            "springer.com",
            "wiley.com",
            "academic.oup.com"
        ],
        "Trusted Health Portal": [
            "cancer.org",
            "cancer.gov",
            "mayoclinic.org",
            "cdc.gov",
            "who.int",
            "webmd.com",
            "healthline.com",
            "medicalnewstoday.com",
            "nccih.nih.gov",
            "cancerresearchuk.org",
            "maxhealthcare.in"
        ],
        "Medical Institution": [
            "clevelandclinic.org",
            "hopkinsmedicine.org",
            "mskcc.org",
            "mdanderson.org",
            "mayo.edu",
            "stanfordhealthcare.org"
        ],
        "Government Health Agency": [
            ".gov/health",
            ".gov/diseases",
            ".gov/about-cancer",
            ".gov/tb",
            "cdc.gov",
            "fda.gov",
            "hhs.gov"
        ]
    }

COUNTRY_TLD_MAP = {
    "us": "United States",
    "uk": "United Kingdom",
    "de": "Germany",
    "ca": "Canada",
    "au": "Australia",
    "in": "India",
    "n/a": "Unknown"
}

# #good models from HuggingFace for medical/general QA
# FREE_MODELS = {
#     "llama_3b": {
#         "name": "meta-llama/Llama-3.2-3B-Instruct",
#         "type": "huggingface",
#         "description": "Meta's Llama 3.2 3B - Good general purpose model"
#     },
#     "phi_3": {
#         "name": "microsoft/Phi-3-mini-4k-instruct",
#         "type": "huggingface",
#         "description": "Microsoft Phi-3 - Efficient small model"
#     },
#     "gemma_2b": {
#         "name": "google/gemma-2-2b-it",
#         "type": "huggingface",
#         "description": "Google Gemma 2 2B - Instruction tuned"
#     },
#     "mistral_7b": {
#         "name": "mistralai/Mistral-7B-Instruct-v0.3",
#         "type": "huggingface",
#         "description": "Mistral 7B - Strong performance"
#     },
#     "zephyr_7b": {
#         "name": "HuggingFaceH4/zephyr-7b-beta",
#         "type": "huggingface",
#         "description": "Zephyr 7B - Fine-tuned for helpfulness"
#     }
# }