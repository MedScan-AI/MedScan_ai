import re
import logging
from enum import Enum
from typing import Tuple

logger = logging.getLogger(__name__)


class QueryStatus(Enum):
    """Status of query evaluation."""
    VALID = "valid"
    UNCLEAR = "unclear"
    OFF_TOPIC = "off_topic"
    HARMFUL = "harmful"
    DIAGNOSIS_SEEKING = "diagnosis_seeking"


class InputGuardrails:
    """Evaluates user queries for a medical QA bot on lung cancer and TB."""
    
    def __init__(self):
        """Initialize the input guardrails."""
        # Medical topics related to lung cancer and TB
        self.medical_keywords = [
            # Lung Cancer terms
            "lung cancer", "nsclc", "sclc", "non-small cell", "small cell",
            "adenocarcinoma", "squamous cell carcinoma", "mesothelioma",
            "lung tumor", "lung nodule", "pulmonary oncology", "thoracic cancer",
            "chemotherapy", "radiation therapy", "immunotherapy", "targeted therapy",
            "egfr", "alk", "kras", "pd-l1", "biopsy", "staging", "metastasis",
            
            # TB terms
            "tuberculosis", "tb", "latent tb", "active tb", "mycobacterium",
            "mtb", "rifampicin", "isoniazid", "dots", "bcg", "mantoux",
            "chest x-ray", "sputum", "pulmonary tb", "extrapulmonary",
            
            # General respiratory/medical terms
            "lung", "pulmonary", "respiratory", "breathing", "cough",
            "hemoptysis", "dyspnea", "chest pain", "pleural", "bronchial",
            "treatment", "medication", "therapy", "symptoms", "screening",
            "risk factors", "prevention", "transmission", "prognosis",
            "side effects", "clinical trial", "pathology", "radiology"
        ]
        
        # Harmful or inappropriate content
        self.harmful_topics = [
            "suicide", "self-harm", "overdose", "kill myself",
            "untested cure", "miracle cure", "guaranteed cure",
            "replace doctor", "instead of treatment", "skip treatment",
            "home remedy only", "natural cure only", "avoid medication"
        ]
        
        # Compile regex patterns
        self.medical_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(kw) for kw in self.medical_keywords) + r')\b',
            re.IGNORECASE
        )
        self.harmful_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(kw) for kw in self.harmful_topics) + r')\b',
            re.IGNORECASE
        )
    
    def evaluate_query(self, query: str) -> Tuple[QueryStatus, str]:
        """
        Evaluate whether a query is appropriate for the medical QA bot.
        
        Args:
            query: The user query to evaluate
            
        Returns:
            Tuple of (QueryStatus, reason/message)
        """
        try:
            logger.info(f"Evaluating query: {query[:100]}...")
            
            # Basic validation
            if not query or len(query.strip()) < 5:
                logger.warning("Query too short")
                return (
                    QueryStatus.UNCLEAR,
                    "Your question is too brief. Please provide more details so I can help clarify information about lung cancer or TB."
                )
            
            # Check for harmful content
            if self.harmful_pattern.search(query):
                logger.warning("Harmful content detected")
                return (
                    QueryStatus.HARMFUL,
                    "I'm here to provide educational information, not medical advice. If you're experiencing a medical emergency or crisis, please contact emergency services or a healthcare professional immediately."
                )
            
            # Check for diagnosis-seeking behavior
            if self.diagnosis_pattern.search(query):
                logger.warning("Diagnosis-seeking query detected")
                return (
                    QueryStatus.DIAGNOSIS_SEEKING,
                    "I cannot diagnose medical conditions or provide personal medical advice. I can only provide general educational information about lung cancer and TB. Please consult a healthcare professional for diagnosis and treatment recommendations."
                )
            
            # Check if query is medically relevant
            if not self.medical_pattern.search(query):
                logger.warning("Off-topic query")
                return (
                    QueryStatus.OFF_TOPIC,
                    "I'm specifically designed to clarify information about lung cancer and tuberculosis (TB). Your question doesn't appear to relate to these topics. Could you rephrase your question to focus on lung cancer or TB?"
                )
            
            # Query passed all checks
            logger.info("Query passed all guardrails")
            return QueryStatus.VALID, ""
            
        except Exception as e:
            logger.error(f"Error evaluating query: {e}")
            return (
                QueryStatus.UNCLEAR,
                "I encountered an issue processing your question. Please try rephrasing it."
            )


# Example usage function
def validate_medical_qa(query: str, response: str) -> Tuple[bool, str, str]:
    """
    Complete validation pipeline for medical QA.
    
    Args:
        query: User input query
        response: Generated response
        
    Returns:
        Tuple of (success, validated_query_status, final_response)
    """
    try:
        # Input validation
        input_guard = InputGuardrails()
        query_status, query_message = input_guard.evaluate_query(query)
        
        if query_status != QueryStatus.VALID:
            logger.warning(f"Query validation failed: {query_status}")
            return False, query_status.value, query_message

        
        # Add safety footer if needed
        footer = (
            "\n\n---\n"
            "**Important:** This information is for educational purposes only and "
            "should not replace professional medical advice. Please consult a "
            "healthcare provider for diagnosis, treatment, or medical guidance."
        )
        final_response = response + footer
        
        logger.info("Complete validation successful")
        return True, "valid", final_response
        
    except Exception as e:
        logger.error(f"Error in validation pipeline: {e}")
        return False, "error", "An error occurred. Please try again."