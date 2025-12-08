"""
gradcam.py - GradCAM (Gradient-weighted Class Activation Mapping) visualization
Based on Keras official example: https://keras.io/examples/vision/grad_cam/
Generates heatmaps showing which regions of an image the model focuses on
"""
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import cv2
from typing import Tuple, Optional
import io
import base64

import logging
logger = logging.getLogger(__name__)


class GradCAM:
    """Generate GradCAM visualizations for TensorFlow/Keras models."""
    
    def __init__(self, model, class_names: list):
        """
        Initialize GradCAM.
        
        Args:
            model: Trained Keras model
            class_names: List of class names
        """
        self.model = model
        self.class_names = class_names
        
        # Find the last convolutional layer
        # Look for Conv2D layers or layers with 4D output
        self.conv_layer = None
        self.conv_layer_name = None
        
        # First, try to find Conv2D layers
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                self.conv_layer = layer
                self.conv_layer_name = layer.name
                break
        
        # If not found, look for any layer with 4D output (excluding input)
        if self.conv_layer is None:
            for layer in reversed(self.model.layers):
                # Skip input layer
                if isinstance(layer, tf.keras.layers.InputLayer):
                    continue
                try:
                    output_shape = layer.output_shape
                    # Check if it's 4D: (batch, height, width, channels)
                    if isinstance(output_shape, (list, tuple)) and len(output_shape) == 4:
                        # Check if it's not the input
                        if hasattr(layer, 'output') and layer.output is not None:
                            self.conv_layer = layer
                            self.conv_layer_name = layer.name
                            break
                except:
                    continue
        
        if self.conv_layer is None:
            raise ValueError("Could not find a suitable convolutional layer for GradCAM")
        
        logger.info(f"Using layer '{self.conv_layer_name}' for GradCAM")
        
        # Build model that outputs both conv layer activations and predictions
        # This is the standard approach from Keras documentation
        self.grad_model = tf.keras.models.Model(
            inputs=self.model.inputs,
            outputs=[self.conv_layer.output, self.model.output]
        )
    
    def make_gradcam_heatmap(
        self,
        img_array: np.ndarray,
        pred_index: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Generate GradCAM heatmap for an image.
        Based on Keras official GradCAM implementation.
        
        Args:
            img_array: Preprocessed image array (1, H, W, C)
            pred_index: Class index to generate heatmap for (None = use predicted class)
            
        Returns:
            Tuple of (heatmap array, predicted class index)
        """
        # Convert to tensor if needed
        if isinstance(img_array, np.ndarray):
            img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        else:
            img_tensor = img_array
        
        # Get predictions and conv layer outputs
        # Use GradientTape to compute gradients
        with tf.GradientTape() as tape:
            # Forward pass: get conv layer outputs and predictions
            conv_outputs, predictions = self.grad_model(img_tensor)
            
            # Determine which class to visualize
            if pred_index is None:
                pred_index = int(tf.argmax(predictions[0]).numpy())
            
            # Get the score for the target class
            # This is what we want to maximize - the prediction for the target class
            class_channel = predictions[:, pred_index]
        
        # Compute gradients of the target class score with respect to conv layer outputs
        # This tells us how much each pixel in the conv layer contributes to the class score
        grads = tape.gradient(class_channel, conv_outputs)
        
        if grads is None:
            logger.error("Gradients are None. Model may have non-differentiable layers.")
            raise ValueError("Could not compute gradients. Check model architecture.")
        
        # Check if gradients are all zeros
        if tf.reduce_sum(tf.abs(grads)) < 1e-8:
            logger.warning("Gradients are very small or zero. Model may not be suitable for GradCAM.")
        
        # Global average pooling of gradients (weights for each feature map channel)
        # This gives us the importance of each channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        logger.debug(f"Pooled grads - min: {tf.reduce_min(pooled_grads):.6f}, "
                    f"max: {tf.reduce_max(pooled_grads):.6f}, "
                    f"mean: {tf.reduce_mean(pooled_grads):.6f}")
        
        # Weight the feature maps by the pooled gradients
        # This is the key step: multiply each channel by its importance
        conv_outputs = conv_outputs[0]  # Remove batch dimension: (H, W, C)
        pooled_grads = pooled_grads  # Shape: (C,)
        
        # Reshape pooled_grads to (1, 1, C) for proper broadcasting with (H, W, C)
        pooled_grads = tf.reshape(pooled_grads, (1, 1, -1))
        
        # Element-wise multiplication: (H, W, C) * (1, 1, C) -> (H, W, C)
        # Then sum across channels to create a 2D heatmap (H, W)
        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
        
        # Normalize heatmap to [0, 1]
        # Use ReLU to ensure non-negative values
        heatmap = tf.maximum(heatmap, 0)
        heatmap_max = tf.reduce_max(heatmap)
        
        # Avoid division by zero
        if heatmap_max > 0:
            heatmap = heatmap / heatmap_max
        else:
            logger.warning("Heatmap is all zeros - model may not be suitable for GradCAM")
        
        # Convert to numpy
        heatmap = heatmap.numpy()
        
        # Verify heatmap has non-zero values
        if np.max(heatmap) < 0.01:
            logger.warning(f"Heatmap values are very low (max={np.max(heatmap):.6f}). "
                         f"GradCAM may not be working correctly.")
        
        return heatmap, pred_index
    
    def generate_visualization(
        self,
        img_array: np.ndarray,
        original_image: Image.Image,
        class_name: str,
        pred_index: int,
        save_path: Optional[str] = None
    ) -> Tuple[bytes, str]:
        """
        Generate three-panel GradCAM visualization.
        
        Args:
            img_array: Preprocessed image array (1, H, W, C)
            original_image: Original PIL Image
            class_name: Name of the predicted class
            pred_index: Predicted class index
            save_path: Optional path to save the image
            
        Returns:
            Tuple of (image bytes, base64 encoded string)
        """
        # Generate heatmap
        heatmap, actual_pred_index = self.make_gradcam_heatmap(img_array, pred_index)
        
        logger.info(f"Generated heatmap - min: {np.min(heatmap):.4f}, max: {np.max(heatmap):.4f}, mean: {np.mean(heatmap):.4f}")
        
        # Resize heatmap to match original image size
        original_size = original_image.size  # (width, height)
        heatmap_resized = cv2.resize(heatmap, original_size, interpolation=cv2.INTER_LINEAR)
        
        # Ensure heatmap is in [0, 1] range after resize
        if heatmap_resized.max() > 0:
            heatmap_resized = heatmap_resized / heatmap_resized.max()
        heatmap_resized = np.clip(heatmap_resized, 0, 1)
        
        # Convert heatmap to RGB colormap (JET colormap: blue=low, red=high)
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        heatmap_colored = heatmap_colored.astype(np.float32) / 255.0
        
        # Convert original image to numpy array
        original_array = np.array(original_image)
        if len(original_array.shape) == 2:  # Grayscale
            original_array = np.stack([original_array] * 3, axis=-1)
        elif original_array.shape[2] == 4:  # RGBA
            original_array = original_array[:, :, :3]
        
        # Normalize original image to [0, 1] if needed
        if original_array.max() > 1.0:
            original_array = original_array.astype(np.float32) / 255.0
        else:
            original_array = original_array.astype(np.float32)
        
        # Create overlay (blend original with heatmap)
        # Higher weight on heatmap to make it more visible
        overlay = 0.4 * original_array + 0.6 * heatmap_colored
        overlay = np.clip(overlay, 0, 1)
        
        # Create three-panel visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Panel 1: Original Image
        axes[0].imshow(original_array)
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Panel 2: GradCAM Heatmap
        im2 = axes[1].imshow(heatmap_resized, cmap='jet', vmin=0, vmax=1)
        axes[1].set_title(f'{class_name} GradCAM Heatmap', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        # Add colorbar
        cbar = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.set_label('Activation Intensity', rotation=270, labelpad=15)
        
        # Panel 3: Overlay
        axes[2].imshow(overlay)
        axes[2].set_title(f'{class_name} GradCAM Overlay', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        image_bytes = buf.read()
        buf.close()
        plt.close(fig)
        
        # Convert to base64
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Save to file if path provided
        if save_path:
            with open(save_path, 'wb') as f:
                f.write(image_bytes)
        
        return image_bytes, image_b64
