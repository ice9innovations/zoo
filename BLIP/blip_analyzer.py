#!/usr/bin/env python3
"""
BLIP Caption Analyzer - Core caption generation functionality

Handles BLIP model loading, image preprocessing, and caption generation
using BLIP (Bootstrapping Language-Image Pre-training) model.
"""

import os
import sys
import logging
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from typing import Dict, Any, Optional

# Add current directory to Python path for model imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)


class BlipAnalyzer:
    """Core BLIP caption analysis functionality"""
    
    def __init__(self, 
                 image_size: int = 384,
                 model_path: str = "./model_base_capfilt_large.pth"):
        """Initialize BLIP analyzer with model configuration"""
        
        self.image_size = image_size
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
        logger.info(f"✅ BlipAnalyzer initialized - Device: {self.device}")
    
    def initialize(self) -> bool:
        """Initialize BLIP model - fail fast if it doesn't work"""
        try:
            # Try to import BLIP model
            try:
                from models.blip import blip_decoder
            except ImportError:
                logger.error("BLIP models not found. Please install BLIP or ensure models directory exists.")
                return False
                
            # Check if model file exists
            if not os.path.exists(self.model_path):
                logger.error(f"BLIP model not found: {self.model_path}")
                logger.error("Please download the required model: model_base_capfilt_large.pth")
                return False
                
            logger.info(f"Loading BLIP model from {self.model_path}")
            
            # Load model with base ViT ('large' refers to caption filtering, not ViT size)
            self.model = blip_decoder(pretrained=self.model_path, image_size=self.image_size, vit='base')
            self.model.eval()
            self.model = self.model.to(self.device)
            
            # Use FP32 for stability (similar to YOLO, BLIP has FP16 compatibility issues)
            logger.info("Using FP32 for BLIP model stability")
            
            logger.info("✅ BLIP model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load BLIP model: {e}")
            return False
    
    def analyze_caption_from_array(self, image_array) -> Dict[str, Any]:
        """
        Generate caption from numpy array (in-memory processing)
        
        Args:
            image_array: Image as numpy array or PIL Image
            
        Returns:
            Dict containing caption generation results
        """
        try:
            # Convert numpy array to PIL Image if needed
            if hasattr(image_array, 'shape'):  # numpy array
                from PIL import Image as PILImage
                image = PILImage.fromarray(image_array)
            else:
                image = image_array  # assume it's already a PIL Image
            
            return self._generate_caption(image)
            
        except Exception as e:
            logger.error(f"BLIP caption analysis error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'caption': ''
            }
    
    def _generate_caption(self, image: Image.Image) -> Dict[str, Any]:
        """Generate caption for PIL Image using BLIP model"""
        if not self.model:
            return {
                'success': False,
                'error': "Model not loaded",
                'caption': ''
            }
            
        try:
            # Preprocess image
            image_tensor = self._preprocess_image(image)
            if image_tensor is None:
                return {
                    'success': False,
                    'error': "Failed to preprocess image", 
                    'caption': ''
                }
                
            # Generate caption with FP16 optimization
            with torch.no_grad():
                # Use autocast for FP16 inference if model supports it
                use_autocast = (self.device.type == 'cuda' and hasattr(self.model, 'dtype') and 
                              self.model.dtype == torch.float16)
                
                # Ensure proper tensor format
                if image_tensor.dim() != 4:
                    logger.error(f"Invalid image tensor dimensions: {image_tensor.shape}")
                    return {
                        'success': False,
                        'error': "Invalid image format",
                        'caption': ''
                    }
                    
                # Try with num_beams=1 first (primary fix for tensor size issues)
                try:
                    if use_autocast:
                        with torch.cuda.amp.autocast():
                            caption = self.model.generate(
                                image_tensor, 
                                sample=False, 
                                num_beams=1,  # Fix for tensor size mismatch issue
                                max_length=20, 
                                min_length=5
                            )
                    else:
                        caption = self.model.generate(
                            image_tensor, 
                            sample=False, 
                            num_beams=1,  # Fix for tensor size mismatch issue
                            max_length=20, 
                            min_length=5
                        )
                except RuntimeError as e:
                    if "size of tensor" in str(e):
                        logger.warning("Beam search failed, trying with sample=True")
                        # Fallback: use sampling instead of beam search
                        if use_autocast:
                            with torch.cuda.amp.autocast():
                                caption = self.model.generate(
                                    image_tensor, 
                                    sample=True, 
                                    num_beams=1,
                                    max_length=20, 
                                    min_length=5
                                )
                        else:
                            caption = self.model.generate(
                                image_tensor, 
                                sample=True, 
                                num_beams=1,
                                max_length=20, 
                                min_length=5
                            )
                    else:
                        raise e
                
            caption_text = caption[0] if caption else "No caption generated"
            logger.info(f"Generated caption: {caption_text}")
            
            return {
                'success': True,
                'error': None,
                'caption': caption_text
            }
            
        except Exception as e:
            logger.error(f"Error generating caption: {e}")
            return {
                'success': False,
                'error': f"Caption generation failed: {str(e)}",
                'caption': ''
            }
    
    def _preprocess_image(self, image: Image.Image) -> Optional[torch.Tensor]:
        """Preprocess PIL Image for BLIP model"""
        try:
            logger.info("Preprocessing image from memory")
            
            # Ensure RGB format (handle different image modes)
            if image.mode != 'RGB':
                logger.info(f"Converting image from {image.mode} to RGB")
                image = image.convert('RGB')
            
            # Log original image dimensions
            logger.info(f"Original image size: {image.size}")
            
            transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ])
            
            image_tensor = transform(image)
            
            # Add batch dimension and move to device
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            logger.info(f"Preprocessed tensor shape: {image_tensor.shape}")
            return image_tensor
            
        except Exception as e:
            logger.error(f"Image preprocessing error: {e}")
            return None
    
    def __del__(self):
        """Cleanup BLIP model resources"""
        if hasattr(self, 'model') and self.model:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()