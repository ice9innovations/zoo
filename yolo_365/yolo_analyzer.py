#!/usr/bin/env python3
"""
Generic YOLO Object Detection Analyzer - Core detection functionality

Handles YOLO model loading, image preprocessing, and object detection
using Ultralytics YOLO models (YOLOv8, YOLOv11, etc.) with support for
custom training datasets (COCO, Object365, OIv7, etc.).
"""

import os
import logging
import torch
import json
import requests
import random
import time
import numpy as np
from PIL import Image
from ultralytics import YOLO
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class YoloAnalyzer:
    """Generic YOLO object detection functionality for any YOLO model"""
    
    def __init__(self,
                 model_path: str = None,
                 model_candidates: List[str] = None,
                 confidence_threshold: float = 0.25,
                 iou_threshold: float = 0.3,
                 max_detections: int = 100,
                 api_host: str = None,
                 api_port: int = None,
                 api_timeout: float = None,
                 service_name: str = "YOLO",
                 dataset: str = None,
                 emoji_mappings: Dict[str, str] = None):
        """
        Initialize YOLO analyzer with model configuration

        Args:
            model_path: Primary model path to try first
            model_candidates: List of fallback models to try if primary fails
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            max_detections: Maximum number of detections per image
            api_host: API host for emoji mappings
            api_port: API port for emoji mappings
            api_timeout: API timeout for emoji mappings
            service_name: Name for logging (e.g., "YOLOv8", "YOLOv11", "YOLO-OIv7")
            dataset: Dataset name (e.g., "COCO", "Object365", "Open Images V7")
            emoji_mappings: Pre-loaded emoji mappings dict
        """
        
        self.model_path = model_path
        self.model_candidates = model_candidates or [
            'yolo11x.pt', 'yolo11l.pt', 'yolo11m.pt', 'yolo11s.pt', 'yolo11n.pt',  # YOLOv11
            'yolov8x.pt', 'yolov8l.pt', 'yolov8m.pt', 'yolov8s.pt', 'yolov8n.pt'   # YOLOv8
        ]
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.service_name = service_name
        self.dataset = dataset
        
        # API configuration for emoji mappings
        self.api_host = api_host or os.getenv('API_HOST')
        self.api_port = int(api_port or os.getenv('API_PORT', 0)) if api_port or os.getenv('API_PORT') else None
        self.api_timeout = float(api_timeout or os.getenv('API_TIMEOUT', 10.0)) if api_timeout or os.getenv('API_TIMEOUT') else 10.0
        
        # Device configuration
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.backends.mps.is_available():
            self.device = 'mps'

        self.model = None
        self.emoji_mappings = emoji_mappings or {}
        self.loaded_model_name = None
        
        logger.info(f"✅ {self.service_name}Analyzer initialized - Device: {self.device}")
    
    def initialize(self) -> bool:
        """Initialize YOLO model and emoji mappings - fail fast if it doesn't work"""
        try:
            # Load emoji mappings only if not provided during initialization
            if not self.emoji_mappings:
                if not self._load_emoji_mappings():
                    logger.warning(f"{self.service_name}: Failed to load emoji mappings, continuing without emojis")
            else:
                logger.info(f"✅ {self.service_name}: Using pre-loaded emoji mappings ({len(self.emoji_mappings)} entries)")

            # Initialize YOLO model
            return self._initialize_yolo_model()

        except Exception as e:
            logger.error(f"❌ Failed to initialize {self.service_name}Analyzer: {e}")
            return False
    
    def _load_emoji_mappings(self) -> bool:
        """Load emoji mappings from GitHub, fall back to local cache"""
        github_url = "https://raw.githubusercontent.com/ice9innovations/animal-farm/refs/heads/main/config/365.json"
        local_cache_path = '365.json'

        auto_update = os.getenv('AUTO_UPDATE', 'true').lower() == 'true'

        if auto_update:
            try:
                import requests
                logger.info(f"🔄 {self.service_name}: Loading emoji mappings from GitHub: {github_url}")
                response = requests.get(github_url, timeout=10.0)
                response.raise_for_status()

                # Save to local cache (preserve emoji characters)
                with open(local_cache_path, 'w', encoding='utf-8') as f:
                    json.dump(response.json(), f, indent=2, ensure_ascii=False)

                self.emoji_mappings = response.json()
                logger.info(f"✅ {self.service_name}: Loaded emoji mappings from GitHub and cached locally ({len(self.emoji_mappings)} entries)")
                return True
            except Exception as e:
                logger.warning(f"⚠️  {self.service_name}: Failed to load emoji mappings from GitHub ({e}), falling back to local cache")

        # Fall back to local cache
        try:
            with open(local_cache_path, 'r') as f:
                self.emoji_mappings = json.load(f)
            logger.info(f"✅ {self.service_name}: Successfully loaded emoji mappings from local cache ({len(self.emoji_mappings)} entries)")
            return True
        except Exception as local_error:
            logger.error(f"❌ {self.service_name}: Failed to load local emoji mappings: {local_error}")
            self.emoji_mappings = {}
            return False
    
    def _load_local_emoji_mappings(self) -> bool:
        """Load emoji mappings from local file"""
        try:
            with open('emoji_mappings.json', 'r') as f:
                self.emoji_mappings = json.load(f)
            logger.info(f"✅ {self.service_name}: Loaded emoji mappings from local file ({len(self.emoji_mappings)} entries)")
            return True
        except Exception as e:
            logger.error(f"❌ {self.service_name}: Failed to load local emoji mappings: {e}")
            logger.warning(f"⚠️ {self.service_name}: No emoji mappings available - emojis will be None")
            self.emoji_mappings = {}
            return False
    
    def _initialize_yolo_model(self) -> bool:
        """Initialize YOLO model with custom training"""
        try:
            # Try primary model path first if provided
            if self.model_path and os.path.exists(self.model_path):
                logger.info(f"Loading custom {self.service_name} model: {self.model_path}")
                self.model = YOLO(self.model_path)
                self.loaded_model_name = os.path.basename(self.model_path)
                logger.info(f"✅ Custom {self.service_name} model loaded successfully")
            else:
                if self.model_path:
                    logger.warning(f"Primary model not found at {self.model_path}, trying fallback models")
                
                # Try fallback models
                model_loaded = False
                for model_file in self.model_candidates:
                    try:
                        logger.info(f"Attempting to load standard {self.service_name} model: {model_file}")
                        self.model = YOLO(model_file)
                        self.loaded_model_name = model_file
                        logger.info(f"✅ Standard {self.service_name} model loaded: {model_file}")
                        model_loaded = True
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load {model_file}: {e}")
                        continue
                
                if not model_loaded:
                    logger.error(f"No {self.service_name} model could be loaded")
                    return False
            
            # Test the model with a dummy prediction
            dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
            test_results = self.model.predict(dummy_image, verbose=False, device=self.device)
            
            logger.info(f"Model device: {self.model.device}")
            logger.info(f"Model precision: FP32")
            logger.info(f"Confidence threshold: {self.confidence_threshold}")
            if self.dataset:
                logger.info(f"Dataset: {self.dataset}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing {self.service_name} model: {e}")
            return False
    
    def get_emoji(self, word: str) -> Optional[str]:
        """Get emoji using direct mapping lookup"""
        if not word:
            return None
        
        word_clean = word.lower().strip().replace(' ', '_').replace('(', '').replace(')', '')
        
        # Try exact match
        if word_clean in self.emoji_mappings:
            return self.emoji_mappings[word_clean]
        
        # Try singular form for common plurals
        if word_clean.endswith('s') and len(word_clean) > 3:
            singular = word_clean[:-1]
            if singular in self.emoji_mappings:
                return self.emoji_mappings[singular]
        
        return None
    
    def check_shiny(self) -> Tuple[bool, int]:
        """Check if this detection should be shiny (1/2500 chance)"""
        roll = random.randint(1, 2500)
        is_shiny = roll == 1
        return is_shiny, roll
    
    def calculate_iou(self, box1: Dict[str, float], box2: Dict[str, float]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        # Extract coordinates
        x1_1, y1_1 = box1['x'], box1['y']
        x2_1, y2_1 = x1_1 + box1['width'], y1_1 + box1['height']
        
        x1_2, y1_2 = box2['x'], box2['y']
        x2_2, y2_2 = x1_2 + box2['width'], y1_2 + box2['height']
        
        # Calculate intersection
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        # Check if there's an intersection
        if x1_inter >= x2_inter or y1_inter >= y2_inter:
            return 0.0
        
        # Calculate intersection area
        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # Calculate union area
        area1 = box1['width'] * box1['height']
        area2 = box2['width'] * box2['height'] 
        union = area1 + area2 - intersection
        
        # Return IoU
        return intersection / union if union > 0 else 0.0
    
    def apply_iou_filtering(self, detections: List[Dict]) -> List[Dict]:
        """Apply IoU-based filtering to merge overlapping detections of the same class"""
        if not detections:
            return detections
        
        # Group detections by class
        class_groups = {}
        for detection in detections:
            class_name = detection.get('class_name', '')
            if class_name not in class_groups:
                class_groups[class_name] = []
            class_groups[class_name].append(detection)
        
        filtered_detections = []
        
        # Process each class separately
        for class_name, class_detections in class_groups.items():
            if len(class_detections) == 1:
                # Only one detection for this class, keep it
                filtered_detections.extend(class_detections)
                continue
            
            # For multiple detections of the same class, apply IoU filtering
            keep_indices = []
            
            for i, det1 in enumerate(class_detections):
                should_keep = True
                
                for j in keep_indices:
                    det2 = class_detections[j]
                    bbox1 = det1.get('bbox', {})
                    bbox2 = det2.get('bbox', {})
                    
                    # Calculate IoU if both have valid bboxes
                    if all(k in bbox1 for k in ['x', 'y', 'width', 'height']) and \
                       all(k in bbox2 for k in ['x', 'y', 'width', 'height']):
                        iou = self.calculate_iou(bbox1, bbox2)
                        
                        if iou > self.iou_threshold:
                            # High overlap detected
                            if det1['confidence'] <= det2['confidence']:
                                # Current detection has lower confidence, don't keep it
                                should_keep = False
                                logger.debug(f"{self.service_name} IoU filter: Removing {class_name} "
                                           f"conf={det1['confidence']:.3f} (IoU={iou:.3f} with "
                                           f"conf={det2['confidence']:.3f})")
                                break
                            else:
                                # Current detection has higher confidence, remove the previous one
                                keep_indices.remove(j)
                                logger.debug(f"{self.service_name} IoU filter: Replacing {class_name} "
                                           f"conf={det2['confidence']:.3f} with "
                                           f"conf={det1['confidence']:.3f} (IoU={iou:.3f})")
                
                if should_keep:
                    keep_indices.append(i)
            
            # Add the kept detections
            for i in keep_indices:
                filtered_detections.append(class_detections[i])
            
            logger.debug(f"{self.service_name} IoU filter: {class_name} {len(class_detections)} → {len(keep_indices)} detections")
        
        return filtered_detections
    
    def process_yolo_results(self, results) -> List[Dict[str, Any]]:
        """Process YOLO model results into structured format"""
        detections = []
        
        if not results or len(results) == 0:
            return detections
            
        result = results[0]
        
        if result.boxes is None or len(result.boxes) == 0:
            return detections
            
        for box in result.boxes:
            try:
                # Get bounding box coordinates
                coords = box.xyxy[0].tolist()
                coords = [round(x) for x in coords]
                x1, y1, x2, y2 = coords
                
                # Get class information
                class_id = int(box.cls[0].item())
                raw_class_name = result.names[class_id] if class_id < len(result.names) else f"class_{class_id}"
                
                # Normalize class name: lowercase, spaces to underscores, handle slashes
                class_name = raw_class_name.lower().replace(' ', '_').replace('/', '_')
                confidence = round(box.conf[0].item(), 3)
                
                # Only include detections above confidence threshold
                if confidence >= self.confidence_threshold:
                    # Look up emoji
                    emoji = self.get_emoji(class_name)
                    
                    detection = {
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": confidence,
                        "bbox": {
                            "x": x1,
                            "y": y1,
                            "width": x2 - x1,
                            "height": y2 - y1
                        },
                        "emoji": emoji
                    }
                    
                    detections.append(detection)
                    
            except Exception as e:
                logger.warning(f"Error processing detection: {e}")
                continue
                
        # Sort by confidence (highest first)
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Apply IoU-based filtering to merge overlapping detections of the same class
        filtered_detections = self.apply_iou_filtering(detections)
        
        # Limit number of detections
        return filtered_detections[:self.max_detections]
    
    def analyze_from_array(self, image_array) -> Dict[str, Any]:
        """
        Analyze objects from numpy array or PIL Image (in-memory processing)
        
        Args:
            image_array: Image as numpy array or PIL Image
            
        Returns:
            Dict containing object detection results
        """
        start_time = time.time()
        
        try:
            if not self.model:
                return {
                    'success': False,
                    'error': "Model not loaded",
                    'detections': [],
                    'processing_time': 0
                }
            
            # Convert numpy array to PIL Image if needed
            if hasattr(image_array, 'shape'):  # numpy array
                image = Image.fromarray(image_array)
            else:
                image = image_array  # assume it's already a PIL Image
            
            # Ensure image is in RGB mode
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            # Get image dimensions
            image_width, image_height = image.size
                
            # Run YOLO detection on PIL Image directly
            logger.debug(f"Running {self.service_name} detection on PIL Image ({image_width}x{image_height})")
            
            # Convert PIL Image to numpy array for YOLO
            image_array = np.array(image)
            
            # Use standard FP32 inference for stability
            results = self.model.predict(
                image_array,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False,
                device=self.device
            )
            
            # Process results
            detections = self.process_yolo_results(results)
            
            logger.debug(f"Detected {len(detections)} objects")
            
            processing_time = round(time.time() - start_time, 3)
            
            return {
                "success": True,
                "detections": detections,
                "total_detections": len(detections),
                "image_dimensions": {
                    "width": image_width,
                    "height": image_height
                },
                "model_info": {
                    "model_name": self.loaded_model_name,
                    "confidence_threshold": self.confidence_threshold,
                    "iou_threshold": self.iou_threshold,
                    "device": str(self.model.device) if hasattr(self.model, 'device') else self.device,
                    **({"dataset": self.dataset} if self.dataset else {})
                },
                "processing_time": processing_time
            }
            
        except Exception as e:
            processing_time = round(time.time() - start_time, 3)
            logger.error(f"Error during {self.service_name} detection: {e}")
            return {
                "success": False,
                "error": f"Detection failed: {str(e)}",
                "processing_time": processing_time
            }
    
    def get_supported_classes(self) -> List[str]:
        """Get list of supported object classes"""
        if self.model and hasattr(self.model, 'names'):
            try:
                return list(self.model.names.values()) if self.model.names else []
            except Exception as e:
                logger.warning(f"Error getting class names from model: {e}")
        return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information consistent with other services"""
        info = {"framework": "Ultralytics YOLO"}
        if self.dataset:
            info["dataset"] = self.dataset
        return info
    
    def __del__(self):
        """Cleanup YOLO model resources"""
        if hasattr(self, 'model') and self.model:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()