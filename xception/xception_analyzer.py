#!/usr/bin/env python3
"""
Xception Image Classification Analyzer - Core classification functionality

Handles Xception model loading, image preprocessing, and classification
using TensorFlow's Xception model for ImageNet classification.
"""

import logging
import time
from typing import Dict, Any, Optional
from PIL import Image
import tensorflow as tf

logger = logging.getLogger(__name__)


class XceptionAnalyzer:
    """Core Xception image classification functionality"""
    
    def __init__(self, 
                 confidence_threshold: float = 0.15,
                 input_size: int = 299):
        """Initialize Xception analyzer with model configuration"""
        
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size
        self.model = None
        
        # Configure TensorFlow GPU memory growth
        self._configure_gpu()
        
        logger.info(f"✅ XceptionAnalyzer initialized - Input size: {input_size}, Threshold: {confidence_threshold}")
    
    def _configure_gpu(self):
        """Configure TensorFlow GPU settings"""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Configured {len(gpus)} GPU(s) for memory growth")
            except RuntimeError as e:
                logger.error(f"GPU memory configuration error: {e}")
    
    def initialize(self) -> bool:
        """Initialize Xception model - fail fast if it doesn't work"""
        try:
            logger.info("Loading Xception model...")
            self.model = tf.keras.applications.Xception(
                weights='imagenet',
                include_top=True,
                input_shape=(self.input_size, self.input_size, 3)
            )
            logger.info("✅ Xception model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load Xception model: {e}")
            return False
    
    def analyze_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Classify image using Xception model (in-memory processing)
        
        Args:
            image: PIL Image object
            
        Returns:
            Dict containing classification results
        """
        start_time = time.time()
        
        if not self.model:
            return {
                "success": False,
                "error": "Xception model not loaded"
            }
        
        try:
            # Ensure image is in RGB mode
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Preprocess image
            img_array = self._preprocess_image(image)
            if img_array is None:
                return {
                    "success": False,
                    "error": "Failed to preprocess image"
                }
            
            # Run classification
            classification_start = time.time()
            predictions = self.model.predict(img_array, verbose=0)
            classification_time = time.time() - classification_start
            
            # Decode predictions with higher top count for better coverage
            decoded_predictions = tf.keras.applications.xception.decode_predictions(
                predictions, top=100
            )[0]
            
            # Process results with deduplication by both emoji and class name
            classifications = self._process_predictions(decoded_predictions)
            
            logger.info(f"Found {len(classifications)} classifications in {classification_time:.2f}s")
            
            # Get image dimensions from PIL Image
            image_width, image_height = image.size
            
            return {
                "success": True,
                "data": {
                    "classifications": classifications,
                    "total_classifications": len(classifications),
                    "image_dimensions": {
                        "width": image_width,
                        "height": image_height
                    },
                    "model_info": {
                        "confidence_threshold": self.confidence_threshold,
                        "classification_time": round(classification_time, 3),
                        "framework": "TensorFlow",
                        "model": "Xception"
                    }
                },
                "processing_time": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Classification processing error: {str(e)}")
            return {
                "success": False,
                "error": f"Classification failed: {str(e)}",
                "processing_time": time.time() - start_time
            }
    
    def _preprocess_image(self, image: Image.Image) -> Optional[tf.Tensor]:
        """Preprocess PIL Image for Xception model"""
        try:
            # Resize PIL Image directly
            image = image.resize((self.input_size, self.input_size), Image.Resampling.LANCZOS)
            
            # Convert PIL Image to array
            img_array = tf.keras.preprocessing.image.img_to_array(image)
            img_array = tf.expand_dims(img_array, axis=0)
            img_array = tf.keras.applications.xception.preprocess_input(img_array)
            
            return img_array
            
        except Exception as e:
            logger.error(f"Error preprocessing PIL image: {e}")
            return None
    
    def _process_predictions(self, decoded_predictions) -> list:
        """Process and deduplicate predictions"""
        classifications = []
        seen_classes = set()
        
        for i, (class_id, class_name, confidence) in enumerate(decoded_predictions):
            if confidence >= self.confidence_threshold:
                # Skip if we've already seen this class
                if class_name in seen_classes:
                    continue
                
                classification = {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(float(confidence), 3),
                    "rank": i + 1
                }
                
                classifications.append(classification)
                seen_classes.add(class_name)
                
        return classifications
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_loaded": self.model is not None,
            "confidence_threshold": self.confidence_threshold,
            "input_size": self.input_size,
            "framework": "TensorFlow",
            "model": "Xception"
        }
    
    def __del__(self):
        """Cleanup model resources"""
        if hasattr(self, 'model') and self.model:
            del self.model