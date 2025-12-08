"""
Vision Inference Pipeline
Provides REST API endpoints for TB and Lung Cancer detection using ResNet models.
"""

__version__ = "1.0.0"

from .model_loader import ModelLoader

__all__ = ['ModelLoader']
