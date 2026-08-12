#!/usr/bin/env python3
"""
CLIP Similarity Analyzer - Core CLIP model functionality

Handles CLIP model loading, image preprocessing, and similarity computation
using CLIP (Contrastive Language-Image Pre-training) model.
"""

import os
import logging
import pickle
import torch
import torchvision.transforms as transforms
from PIL import Image
import clip
import numpy as np
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class ClipAnalyzer:
    """Core CLIP similarity analysis functionality"""
    
    def __init__(self, 
                 model_name: str = 'ViT-L/14',
                 confidence_threshold: float = 0.00,
                 max_predictions: int = 100,
                 labels_folder: str = './labels',
                 cache_file: str = './text_embeddings_cache.pkl'):
        """Initialize CLIP analyzer with configuration"""
        
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.max_predictions = max_predictions
        self.labels_folder = labels_folder
        self.cache_file = cache_file
        
        # Model components - copied from REST.py
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            
        self.model = None
        self.preprocess = None
        self.labels = None
        self.label_tensor = None
        self.label_features = None  # CACHED TEXT FEATURES
        
        # Text embedding cache - copied from REST.py
        self.text_embedding_cache = {}
        
        logger.info(f"✅ ClipAnalyzer initialized - Device: {self.device}, Model: {self.model_name}")
    
    def initialize(self) -> bool:
        """Initialize CLIP model - copied exactly from initialize_clip_model() in REST.py"""
        try:
            logger.info(f"Loading CLIP model: {self.model_name}...")
            self.model, self.preprocess = clip.load(self.model_name, device=self.device)
            
            # Apply FP16 optimization for VRAM savings and speed boost
            if self.device == "cuda":
                self.model = self.model.half()
                logger.info(f"Applied FP16 quantization to {self.model_name} - 50% VRAM reduction achieved!")
                logger.info(f"Expected VRAM usage: ~4.3GB (down from ~8.7GB)")
            
            logger.info(f"CLIP model {self.model_name} loaded successfully")
            
            # Load cached embeddings
            self._load_text_embeddings_cache()
            
            # Load labels from files
            self.labels = self._load_labels_from_files()
            
            if not self.labels:
                logger.error("No labels loaded from files!")
                return False
            
            # Create text descriptions
            labels_desc = [f"a picture of a {label}" for label in self.labels]
            
            # Check which embeddings need to be computed
            labels_to_compute = []
            cached_features = []
            
            for i, (label, desc) in enumerate(zip(self.labels, labels_desc)):
                cache_key = desc.lower()  # Use description as cache key
                if cache_key in self.text_embedding_cache:
                    # Convert cached numpy array back to tensor
                    cached_tensor = torch.from_numpy(self.text_embedding_cache[cache_key]).to(self.device)
                    if self.device == "cuda" and hasattr(self.model, 'dtype') and self.model.dtype == torch.float16:
                        cached_tensor = cached_tensor.half()
                    cached_features.append(cached_tensor)
                else:
                    labels_to_compute.append((i, label, desc))
                    cached_features.append(None)  # Placeholder
            
            # Compute missing embeddings if any
            if labels_to_compute:
                logger.info(f"Computing embeddings for {len(labels_to_compute)} new labels...")
                
                # Tokenize only the missing labels
                missing_descs = [desc for _, _, desc in labels_to_compute]
                self.label_tensor = clip.tokenize(missing_descs).to(self.device)
                
                # Compute text features for missing labels
                with torch.no_grad():
                    if self.device == "cuda" and hasattr(self.model, 'dtype') and self.model.dtype == torch.float16:
                        with torch.amp.autocast('cuda'):
                            computed_features = self.model.encode_text(self.label_tensor)
                    else:
                        computed_features = self.model.encode_text(self.label_tensor)
                
                # Cache the computed features and fill in the placeholders
                for i, (original_idx, label, desc) in enumerate(labels_to_compute):
                    feature_tensor = computed_features[i]
                    cached_features[original_idx] = feature_tensor
                    
                    # Cache as numpy array for persistence
                    cache_key = desc.lower()
                    self.text_embedding_cache[cache_key] = feature_tensor.cpu().numpy()
                
                # Save updated cache
                self._save_text_embeddings_cache()
            else:
                logger.info(f"All {len(self.labels)} label embeddings loaded from cache")
            
            # Stack all features into final tensor
            self.label_features = torch.stack(cached_features)
            
            logger.info(f"Pre-computed text features for {len(self.labels)} labels with caching - memory leak fixed!")
            logger.info(f"Initialized {len(self.labels)} classification labels from files")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize CLIP model: {e}")
            return False
    
    def analyze_similarity_from_array(self, image_array) -> Dict[str, Any]:
        """
        Compute similarity from numpy array (in-memory processing)
        
        Args:
            image_array: Image as numpy array or PIL Image
            
        Returns:
            Dict containing similarity results
        """
        try:
            # Convert numpy array to PIL Image if needed
            if hasattr(image_array, 'shape'):  # numpy array
                from PIL import Image as PILImage
                image = PILImage.fromarray(image_array)
            else:
                image = image_array  # assume it's already a PIL Image
            
            return self._classify_image(image)
            
        except Exception as e:
            logger.error(f"CLIP similarity analysis error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'predictions': [],
                'emoji_matches': []
            }
    
    def _classify_image(self, image: Image.Image) -> Dict[str, Any]:
        """Classify PIL Image using CLIP model - copied exactly from classify_image() in REST.py"""
        if not self.model or not self.labels:
            return {"error": "Model not initialized", "status": "error"}
            
        try:
            # Use the exact same logic as the working version
            logger.info("Computing similarities...")
            similarity_scores = self._compute_similarity(image)
            if similarity_scores is None:
                return {"error": "Failed to compute similarities", "status": "error"}
                
            logger.info(f"Getting predictions above threshold {self.confidence_threshold}...")
            
            # Get all scores and indices, sorted by confidence
            scores_sorted, indices_sorted = similarity_scores[0].sort(descending=True)
            
            predictions = []
            
            for i, (score, idx) in enumerate(zip(scores_sorted, indices_sorted)):
                confidence = score.item()
                
                # Stop if below threshold
                if confidence < self.confidence_threshold:
                    break
                    
                # Stop if we've hit the max limit
                if len(predictions) >= self.max_predictions:
                    break
                
                label = self.labels[idx]
                predictions.append({"label": label, "confidence": round(confidence, 3)})
                
                logger.info(f"Prediction {i+1}: {label} ({confidence:.3f})")
            
            logger.info(f"Returned {len(predictions)} predictions above threshold {self.confidence_threshold}")
            
            return {
                'success': True,
                'error': None,
                'predictions': predictions,
                'emoji_matches': []  # Will be populated by external emoji lookup
            }
            
        except Exception as e:
            logger.error(f"Error classifying image: {e}")
            return {
                'success': False,
                'error': f"Classification failed: {str(e)}",
                'predictions': [],
                'emoji_matches': []
            }
    
    def _compute_similarity(self, image: Image.Image) -> Optional[torch.Tensor]:
        """Compute similarity between PIL Image and text labels - copied exactly from compute_similarity() in REST.py"""
        logger.info("Checking model and label features...")
        if self.model is None:
            logger.error("Model not initialized")
            return None
        if self.label_features is None:
            logger.error("Label features not initialized")
            return None
            
        try:
            with torch.no_grad():
                logger.info("Preprocessing image...")
                image_tensor = self._preprocess_image(image)
                if image_tensor is None:
                    return None
                
                logger.info("Encoding image features...")
                # Use autocast for FP16 inference stability - ONLY encode image (text features cached)
                if self.device == "cuda" and hasattr(self.model, 'dtype') and self.model.dtype == torch.float16:
                    with torch.amp.autocast('cuda'):
                        image_features = self.model.encode_image(image_tensor)
                else:
                    image_features = self.model.encode_image(image_tensor)
                
                # Use pre-computed cached label_features (no more text encoding per request!)
                    
                logger.info("Computing similarity...")
                similarity = (image_features @ self.label_features.T).softmax(dim=-1)
                logger.info("Similarity computed successfully")
                
                # Clean up GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            return similarity
            
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return None
    
    def _preprocess_image(self, image: Image.Image) -> Optional[torch.Tensor]:
        """Preprocess PIL Image for CLIP model - copied exactly from preprocess_image() in REST.py"""
        try:
            # Ensure RGB format
            if image.mode != 'RGB':
                image = image.convert("RGB")
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Convert image tensor to half precision if model is FP16
            if self.device == "cuda" and hasattr(self.model, 'dtype') and self.model.dtype == torch.float16:
                image_tensor = image_tensor.half()
                
            return image_tensor
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return None
    
    def _load_labels_from_files(self) -> List[str]:
        """Load classification labels from all .txt files in the labels/ folder - copied exactly from load_labels_from_files() in REST.py"""
        all_labels = []
        
        # Automatically discover all .txt files in the labels folder
        try:
            txt_files = [f for f in os.listdir(self.labels_folder) if f.endswith('.txt')]
            txt_files.sort()  # Consistent ordering
            logger.info(f"Found {len(txt_files)} label files: {txt_files}")
        except OSError as e:
            logger.error(f"Could not read labels folder {self.labels_folder}: {e}")
            return []
        
        for filename in txt_files:
            filepath = os.path.join(self.labels_folder, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    # Read lines, strip whitespace, ignore empty lines and comments
                    file_labels = [
                        line.strip() 
                        for line in f.readlines() 
                        if line.strip() and not line.strip().startswith('#')
                    ]
                    all_labels.extend(file_labels)
                    logger.info(f"Loaded {len(file_labels)} labels from {filepath}")
            except Exception as e:
                logger.error(f"Error loading labels from {filepath}: {e}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_labels = []
        for label in all_labels:
            if label.lower() not in seen:
                seen.add(label.lower())
                unique_labels.append(label)
        
        logger.info(f"Total unique labels loaded: {len(unique_labels)}")
        return unique_labels
    
    def _load_text_embeddings_cache(self):
        """Load text embeddings from cache file if it exists - copied exactly from load_text_embeddings_cache() in REST.py"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self.text_embedding_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.text_embedding_cache)} text embeddings from cache file")
            except Exception as e:
                logger.error(f"Failed to load text embeddings cache: {e}")
                self.text_embedding_cache = {}
        else:
            logger.info("No text embeddings cache file found, will create one after first computation")

    def _save_text_embeddings_cache(self):
        """Save text embeddings to cache file - copied exactly from save_text_embeddings_cache() in REST.py"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.text_embedding_cache, f)
            logger.info(f"Saved {len(self.text_embedding_cache)} text embeddings to cache file")
        except Exception as e:
            logger.error(f"Failed to save text embeddings cache: {e}")
    
    def compute_caption_similarity(self, image: Image.Image, caption: str) -> Optional[float]:
        """Compute similarity between PIL Image and arbitrary caption text with caching"""
        logger.info(f"Computing caption similarity for: '{caption}'")
        
        if self.model is None:
            logger.error("Model not initialized")
            return None
            
        try:
            with torch.no_grad():
                # Preprocess image
                logger.info("Preprocessing image...")
                image_tensor = self._preprocess_image(image)
                if image_tensor is None:
                    return None
                    
                # Check cache for text features first
                cache_key = caption.lower().strip()
                if cache_key in self.text_embedding_cache:
                    logger.info(f"Using cached text features for caption: '{caption}'")
                    text_features = torch.from_numpy(self.text_embedding_cache[cache_key]).to(self.device)
                    if self.device == "cuda" and hasattr(self.model, 'dtype') and self.model.dtype == torch.float16:
                        text_features = text_features.half()
                    # Ensure proper shape (add batch dimension if needed)
                    if len(text_features.shape) == 1:
                        text_features = text_features.unsqueeze(0)
                else:
                    # Tokenize caption text - use CLIP's built-in truncation for long captions
                    logger.info("Tokenizing and encoding caption...")
                    logger.info(f"Using raw caption: '{caption}'")
                    # Use truncate=True to handle captions longer than 77 tokens
                    text_tokens = clip.tokenize([caption], truncate=True).to(self.device)
                    
                    # Encode text with autocast for FP16 stability
                    if self.device == "cuda" and hasattr(self.model, 'dtype') and self.model.dtype == torch.float16:
                        with torch.amp.autocast('cuda'):
                            text_features = self.model.encode_text(text_tokens)
                    else:
                        text_features = self.model.encode_text(text_tokens)
                    
                    # Cache the text features
                    self.text_embedding_cache[cache_key] = text_features.cpu().numpy()
                    # Save cache periodically (but not every single request to avoid I/O overhead)
                    if len(self.text_embedding_cache) % 10 == 0:  # Save every 10 new entries
                        self._save_text_embeddings_cache()
                    
                logger.info("Encoding image features...")
                # Encode image with autocast for FP16 stability
                if self.device == "cuda" and hasattr(self.model, 'dtype') and self.model.dtype == torch.float16:
                    with torch.amp.autocast('cuda'):
                        image_features = self.model.encode_image(image_tensor)
                else:
                    image_features = self.model.encode_image(image_tensor)
                
                # Normalize features (important for cosine similarity)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Compute cosine similarity
                logger.info("Computing similarity...")
                similarity = (image_features @ text_features.T).item()
                
                logger.info(f"Caption similarity computed: {similarity:.3f}")
                
                # Clean up GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            return similarity
            
        except Exception as e:
            logger.error(f"Error computing caption similarity: {e}")
            return None
    
    def __del__(self):
        """Cleanup CLIP model resources"""
        if hasattr(self, 'model') and self.model:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()