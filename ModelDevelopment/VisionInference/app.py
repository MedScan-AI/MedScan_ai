"""
app.py - FastAPI application for Vision model inference
Provides endpoints for TB and Lung Cancer detection using ResNet models
"""
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import io

import numpy as np
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from model_loader import ModelLoader
from gradcam import GradCAM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Initialize FastAPI app
app = FastAPI(
    title="Vision Inference API",
    description="REST API for TB and Lung Cancer detection using ResNet models",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model loader
# Supports both local filesystem and GCS
# For GCS: looks in gs://bucket/vision/trained_models/{BUILD_ID}/models/models/YYYY/MM/DD/timestamp/
MODELS_BASE_PATH = os.getenv("MODELS_BASE_PATH", None)
GCS_BUCKET = os.getenv("GCS_BUCKET_NAME", "medscan-pipeline-medscanai-476500")
GCS_PREFIX = os.getenv("GCS_MODELS_PREFIX", "vision/trained_models")
USE_GCS = os.getenv("USE_GCS_MODELS", "true").lower() == "true"  # Default to GCS in Cloud Run

model_loader = ModelLoader(
    models_base_path=MODELS_BASE_PATH,
    gcs_bucket=GCS_BUCKET if USE_GCS else None,
    gcs_prefix=GCS_PREFIX if USE_GCS else None
)

# Load models at startup
tb_loaded = False
lung_cancer_loaded = False


# Pydantic models for request/response
class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    tb_model_loaded: bool
    lung_cancer_model_loaded: bool


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    status: str
    model: str
    model_path: Optional[str] = None
    predicted_class: str
    confidence: float
    class_probabilities: Dict[str, float]
    all_predictions: list
    gradcam_image: Optional[str] = None  # Base64 encoded GradCAM visualization


class ErrorResponse(BaseModel):
    """Response model for errors."""
    error: str
    status: str


def preprocess_image(image: Image.Image, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Preprocess image for model inference.
    
    Args:
        image: PIL Image object
        target_size: Target image size (height, width)
        
    Returns:
        Preprocessed image array ready for model input
    """
    # Resize image
    image = image.resize(target_size, Image.LANCZOS)
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array and normalize
    img_array = np.array(image, dtype=np.float32)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def predict_with_model(model, image_array: np.ndarray, class_names: list = None) -> Dict[str, Any]:
    """
    Run prediction with model.
    
    Args:
        model: TensorFlow model
        image_array: Preprocessed image array
        class_names: Optional list of class names
        
    Returns:
        Dictionary with predictions
    """
    try:
        predictions = model.predict(image_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Get all class probabilities
        class_probs = {}
        if class_names and len(class_names) == len(predictions[0]):
            for i, class_name in enumerate(class_names):
                class_probs[class_name] = float(predictions[0][i])
        else:
            # Use generic class names
            for i, prob in enumerate(predictions[0]):
                class_probs[f"class_{i}"] = float(prob)
        
        predicted_class = class_names[predicted_class_idx] if class_names else f"class_{predicted_class_idx}"
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "class_probabilities": class_probs,
            "all_predictions": [float(p) for p in predictions[0]]
        }
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        raise


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """Load PIL Image from bytes."""
    return Image.open(io.BytesIO(image_bytes))


@app.on_event("startup")
async def startup_event():
    """Load models at startup."""
    global tb_loaded, lung_cancer_loaded
    
    logger.info("Loading models...")
    tb_loaded, lung_cancer_loaded = model_loader.load_all_models()
    
    if tb_loaded:
        logger.info("✓ TB model loaded successfully")
    else:
        logger.warning("✗ TB model failed to load")
    
    if lung_cancer_loaded:
        logger.info("✓ Lung Cancer model loaded successfully")
    else:
        logger.warning("✗ Lung Cancer model failed to load")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        tb_model_loaded=tb_loaded,
        lung_cancer_model_loaded=lung_cancer_loaded
    )


@app.post("/predict/tb", response_model=PredictionResponse)
async def predict_tb(
    file: UploadFile = File(...)
):
    """
    Predict TB from chest X-ray image.
    
    Accepts:
        - Multipart form data with 'file' field (image file)
    
    Returns:
        JSON with prediction results
    """
    if not tb_loaded:
        raise HTTPException(
            status_code=503,
            detail="TB model not loaded"
        )
    
    try:
        # Read file contents
        contents = await file.read()
        image = load_image_from_bytes(contents)
        
        # Keep original image for visualization
        original_image = image.copy()
        
        # Preprocess image
        image_array = preprocess_image(image)
        
        # Get model and class names
        model = model_loader.get_tb_model()
        class_names = model_loader.get_tb_class_names()
        
        # Predict
        result = predict_with_model(model, image_array, class_names)
        
        # Generate GradCAM visualization
        try:
            # Find predicted class index
            pred_class_idx = 0
            if result["predicted_class"] in class_names:
                pred_class_idx = class_names.index(result["predicted_class"])
            else:
                # Find index from class_probabilities
                pred_class_idx = list(result["class_probabilities"].values()).index(result["confidence"])
            
            gradcam = GradCAM(model, class_names)
            _, gradcam_b64 = gradcam.generate_visualization(
                img_array=image_array,
                original_image=original_image,
                class_name=result["predicted_class"],
                pred_index=pred_class_idx
            )
            result["gradcam_image"] = gradcam_b64
        except Exception as e:
            logger.warning(f"Failed to generate GradCAM visualization: {e}", exc_info=True)
            result["gradcam_image"] = None
        
        return PredictionResponse(
            status="success",
            model="tb_CNN_ResNet18",
            model_path=str(model_loader.tb_model_path) if model_loader.tb_model_path else None,
            **result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in TB prediction: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error during prediction: {str(e)}"
        )


@app.post("/predict/lung_cancer", response_model=PredictionResponse)
async def predict_lung_cancer(
    file: UploadFile = File(...)
):
    """
    Predict Lung Cancer from CT scan image.
    
    Accepts:
        - Multipart form data with 'file' field (image file)
    
    Returns:
        JSON with prediction results
    """
    if not lung_cancer_loaded:
        raise HTTPException(
            status_code=503,
            detail="Lung Cancer model not loaded"
        )
    
    try:
        # Read file contents
        contents = await file.read()
        image = load_image_from_bytes(contents)
        
        # Keep original image for visualization
        original_image = image.copy()
        
        # Preprocess image
        image_array = preprocess_image(image)
        
        # Get model and class names
        model = model_loader.get_lung_cancer_model()
        class_names = model_loader.get_lung_cancer_class_names()
        
        # Predict
        result = predict_with_model(model, image_array, class_names)
        
        # Generate GradCAM visualization
        try:
            # Find predicted class index
            pred_class_idx = 0
            if result["predicted_class"] in class_names:
                pred_class_idx = class_names.index(result["predicted_class"])
            else:
                # Find index from class_probabilities
                pred_class_idx = list(result["class_probabilities"].values()).index(result["confidence"])
            
            gradcam = GradCAM(model, class_names)
            _, gradcam_b64 = gradcam.generate_visualization(
                img_array=image_array,
                original_image=original_image,
                class_name=result["predicted_class"],
                pred_index=pred_class_idx
            )
            result["gradcam_image"] = gradcam_b64
        except Exception as e:
            logger.warning(f"Failed to generate GradCAM visualization: {e}", exc_info=True)
            result["gradcam_image"] = None
        
        return PredictionResponse(
            status="success",
            model="lung_cancer_ct_scan_CNN_ResNet18",
            model_path=str(model_loader.lung_cancer_model_path) if model_loader.lung_cancer_model_path else None,
            **result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in Lung Cancer prediction: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error during prediction: {str(e)}"
        )


if __name__ == '__main__':
    import uvicorn
    
    # Get port from environment or use default
    port = int(os.getenv('PORT', 5000))
    host = os.getenv('HOST', '0.0.0.0')
    
    logger.info(f"Starting Vision Inference Server on {host}:{port}")
    logger.info(f"API Documentation available at: http://{host}:{port}/docs")
    logger.info(f"Endpoints available:")
    logger.info(f"  - GET  /health")
    logger.info(f"  - POST /predict/tb")
    logger.info(f"  - POST /predict/lung_cancer")
    
    uvicorn.run(app, host=host, port=port)
