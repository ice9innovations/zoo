#!/usr/bin/env python3
"""
Detectron2 Object Detection Analyzer - Core detection functionality

Handles object detection and instance segmentation using Facebook AI Research's
Detectron2 framework with enhanced features for the standalone detection service.
"""

import os
import logging
import time
import threading
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image

# Detectron2 imports
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

# Local predictor module (exists in Detectron2 repo)
from predictor import VisualizationDemo

logger = logging.getLogger(__name__)


class Detectron2Analyzer:
    """Core Detectron2 object detection and instance segmentation functionality"""
    
    def __init__(self, 
                 config_file: str,
                 confidence_threshold: float = 0.5,
                 coco_classes: List[str] = None,
                 use_half_precision: bool = True):
        """Initialize Detectron2 analyzer"""
        
        self.config_file = config_file
        self.confidence_threshold = confidence_threshold
        self.coco_classes = coco_classes or []
        self.use_half_precision = use_half_precision
        self.demo = None
        
        # Thread lock for model inference
        self.model_lock = threading.Lock()
        
        # IoU threshold for filtering overlapping detections
        self.iou_threshold = 0.3
        
        # Initialize the model
        self._initialize_model()
        
        logger.info("✅ Detectron2Analyzer initialized successfully")
    
    def _initialize_model(self):
        """Initialize Detectron2 model"""
        try:
            # Setup configuration
            cfg = self._setup_cfg()
            if cfg is None:
                raise RuntimeError("Failed to setup Detectron2 configuration")
                
            # Setup logger
            setup_logger(name="fvcore")
            
            # Initialize demo
            self.demo = VisualizationDemo(cfg)
            
            logger.info("Detectron2 model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Detectron2 model: {e}")
            raise
    
    def _setup_cfg(self) -> Any:
        """Setup Detectron2 configuration"""
        try:
            cfg = get_cfg()
            
            # Load config file
            cfg.merge_from_file(self.config_file)
            
            # Set model weights - use local file if available, otherwise download
            local_model_path = "./model_final_280758.pkl"
            if os.path.exists(local_model_path):
                cfg.MODEL.WEIGHTS = local_model_path
                logger.info(f"Using cached local model: {local_model_path}")
            else:
                cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"
                logger.info("Downloading model weights (this may take time on first run)")

            # Set confidence thresholds
            cfg.MODEL.RETINANET.SCORE_THRESH_TEST = self.confidence_threshold
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold
            cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = self.confidence_threshold
            
            # Performance optimizations
            cfg.TEST.DETECTIONS_PER_IMAGE = 20           # Limit to 20 detections max
            cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # Reduce batch size
            
            # Set device - require CUDA
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available. GPU is required for Detectron2.")
            cfg.MODEL.DEVICE = "cuda"
            logger.info(f"Using device: {cfg.MODEL.DEVICE}")
     
            cfg.freeze()
            return cfg
        except Exception as e:
            logger.error(f"Failed to setup config: {e}")
            return None
    
    def analyze_from_array(self, image_array: np.ndarray) -> Dict[str, Any]:
        """
        Analyze objects from numpy array (in-memory processing)
        
        Args:
            image_array: RGB image as numpy array
            
        Returns:
            Dict containing object detection results
        """
        try:
            if not self.demo:
                raise RuntimeError("Detectron2 model not initialized")
            
            start_time = time.time()
            
            # Convert RGB to BGR for Detectron2 (numpy slice - no CV2 needed!)
            img = image_array[:, :, ::-1]
            
            # Run detection with autocast for FP16 optimization (thread-safe)
            detection_start = time.time()
            import torch
            
            # Acquire lock for thread-safe model inference
            with self.model_lock:
                if self.use_half_precision:
                    with torch.amp.autocast('cuda'):
                        predictions, visualized_output = self.demo.run_on_image(img)
                        precision_used = "FP16"
                else:
                    predictions, visualized_output = self.demo.run_on_image(img)
                    precision_used = "FP32"
            
            detection_time = time.time() - detection_start
            
            # Process results (no scaling needed as we're working with original image)
            detections = self._process_detections(predictions, 1.0, 1.0)
            
            logger.info(f"Detected {len(detections)} objects in {detection_time:.2f}s")
            
            # Get image dimensions
            height, width = image_array.shape[:2]
            
            return {
                'detections': detections,
                'total_detections': len(detections),
                'image_dimensions': {
                    'width': width,
                    'height': height
                },
                'model_info': {
                    'confidence_threshold': self.confidence_threshold,
                    'detection_time': round(detection_time, 3),
                    'framework': 'Detectron2',
                    'precision': precision_used
                },
                'processing_time': round(time.time() - start_time, 3)
            }
            
        except Exception as e:
            logger.error(f"Detectron2 analysis error: {str(e)}")
            raise
    
    def analyze_from_pil_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Analyze objects from PIL Image with optional resizing for performance
        
        Args:
            image: PIL Image object
            
        Returns:
            Dict containing object detection results
        """
        try:
            # Store original dimensions before resizing
            original_width, original_height = image.size
            
            # Resize PIL image for faster inference
            resized_image = self._resize_pil_image_for_inference(image, max_size=512)
            resized_width, resized_height = resized_image.size
            
            # Calculate scaling factors to convert coordinates back to original image
            scale_x = original_width / resized_width
            scale_y = original_height / resized_height
            
            # Convert PIL Image to numpy array for Detectron2
            img_array = np.array(resized_image)
            
            # Process with scaling for coordinate adjustment
            result = self.analyze_from_array(img_array)
            
            # Adjust coordinates back to original image scale
            if 'detections' in result:
                for detection in result['detections']:
                    if 'bbox' in detection:
                        bbox = detection['bbox']
                        bbox['x'] = round(bbox['x'] * scale_x)
                        bbox['y'] = round(bbox['y'] * scale_y)
                        bbox['width'] = round(bbox['width'] * scale_x)
                        bbox['height'] = round(bbox['height'] * scale_y)
            
            # Update image dimensions to original
            result['image_dimensions'] = {
                'width': original_width,
                'height': original_height
            }
            
            return result
            
        except Exception as e:
            logger.error(f"PIL Image analysis error: {str(e)}")
            raise
    
    def _resize_pil_image_for_inference(self, image: Image.Image, max_size=512) -> Image.Image:
        """Resize PIL image to speed up inference"""
        width, height = image.size
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return image
    
    def _process_detections(self, predictions: Dict[str, Any], scale_x: float = 1.0, scale_y: float = 1.0) -> List[Dict[str, Any]]:
        """Process Detectron2 predictions into structured format"""
        detections = []
        
        if not predictions or "instances" not in predictions:
            logger.warning("No instances in predictions")
            return detections
            
        instances = predictions["instances"]
        
        # Access detectron2 instances attributes properly
        pred_classes = instances.pred_classes if hasattr(instances, 'pred_classes') else None
        scores = instances.scores if hasattr(instances, 'scores') else None
        
        if pred_classes is None or scores is None:
            logger.warning("Missing pred_classes or scores")
            return detections
        
        # Get bounding boxes if available
        boxes = getattr(instances, "pred_boxes", None)
        
        for i in range(len(pred_classes)):
            # Extract class ID and confidence with specific error handling
            try:
                class_id = int(pred_classes[i].item())
            except (ValueError, TypeError, AttributeError) as e:
                logger.warning(f"Detection {i}: Failed to extract class_id: {e}")
                continue
                
            try:
                confidence = float(scores[i].item())
            except (ValueError, TypeError, AttributeError) as e:
                logger.warning(f"Detection {i}: Failed to extract confidence: {e}")
                continue
                
            # Get class name with +1 offset: COCO classes file has "background" at index 0,
            # but Detectron2 models only predict object classes (0-79), so model ID 0 = "person" at index 1
            if 0 <= class_id + 1 < len(self.coco_classes):
                class_name = self.coco_classes[class_id + 1]
            else:
                class_name = f"class_{class_id}"
                
            # Only include detections above confidence threshold
            if confidence >= self.confidence_threshold:
                detection = {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(confidence, 3)
                }
                
                # Extract bounding box - try multiple approaches to always get spatial info
                bbox_extracted = False
                if boxes is not None and i < len(boxes):
                    # Method 1: Standard tensor extraction
                    try:
                        box = boxes[i].tensor.cpu().numpy()[0]
                        
                        if len(box) >= 4:  # Accept any format with at least 4 coordinates
                            x1_scaled = round(float(box[0]) * scale_x)
                            y1_scaled = round(float(box[1]) * scale_y)
                            x2_scaled = round(float(box[2]) * scale_x)
                            y2_scaled = round(float(box[3]) * scale_y)
                            
                            # Ensure coordinates are in correct order and bounds
                            x1_scaled = max(0, min(x1_scaled, x2_scaled))
                            y1_scaled = max(0, min(y1_scaled, y2_scaled))
                            x2_scaled = max(x1_scaled + 1, x2_scaled)  # Ensure width > 0
                            y2_scaled = max(y1_scaled + 1, y2_scaled)  # Ensure height > 0
                            
                            detection["bbox"] = {
                                "x": x1_scaled,
                                "y": y1_scaled,
                                "width": x2_scaled - x1_scaled,
                                "height": y2_scaled - y1_scaled
                            }
                            bbox_extracted = True
                            
                    except Exception as e:
                        logger.debug(f"Detection {i}: Standard bbox extraction failed: {e}")
                    
                    # Method 2: Try alternative tensor access if Method 1 failed
                    if not bbox_extracted:
                        try:
                            # Try different tensor access patterns
                            if hasattr(boxes[i], 'tensor'):
                                tensor_data = boxes[i].tensor.cpu().numpy()
                                if len(tensor_data.shape) > 1:
                                    box = tensor_data.flatten()[:4]  # Take first 4 values
                                else:
                                    box = tensor_data[:4]
                            else:
                                # Direct numpy array access
                                box = boxes[i].cpu().numpy()[:4]
                            
                            if len(box) >= 4:
                                x1_scaled = round(float(box[0]) * scale_x)
                                y1_scaled = round(float(box[1]) * scale_y) 
                                x2_scaled = round(float(box[2]) * scale_x)
                                y2_scaled = round(float(box[3]) * scale_y)
                                
                                x1_scaled = max(0, min(x1_scaled, x2_scaled))
                                y1_scaled = max(0, min(y1_scaled, y2_scaled))
                                x2_scaled = max(x1_scaled + 1, x2_scaled)
                                y2_scaled = max(y1_scaled + 1, y2_scaled)
                                
                                detection["bbox"] = {
                                    "x": x1_scaled,
                                    "y": y1_scaled,
                                    "width": x2_scaled - x1_scaled,
                                    "height": y2_scaled - y1_scaled
                                }
                                bbox_extracted = True
                                
                        except Exception as e:
                            logger.debug(f"Detection {i}: Alternative bbox extraction failed: {e}")
                    
                    if not bbox_extracted:
                        logger.info(f"Detection {i} ({class_name}): No bounding box could be extracted - complex segmentation shape")
                
                detections.append(detection)
                
        # Sort by confidence (highest first)
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Apply IoU-based filtering to merge overlapping detections of the same class
        filtered_detections = self._apply_iou_filtering(detections)
        
        return filtered_detections
    
    def _calculate_iou(self, box1: Dict[str, float], box2: Dict[str, float]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        # Extract coordinates - now using consistent x,y,width,height format
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
        
        # Calculate areas using width/height from bbox
        area1 = box1['width'] * box1['height']
        area2 = box2['width'] * box2['height']
        
        union = area1 + area2 - intersection
        
        # Return IoU
        return intersection / union if union > 0 else 0.0
    
    def _apply_iou_filtering(self, detections: List[Dict], iou_threshold: float = None) -> List[Dict]:
        """Apply IoU-based filtering to merge overlapping detections of the same class"""
        if iou_threshold is None:
            iou_threshold = self.iou_threshold
            
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
                        iou = self._calculate_iou(bbox1, bbox2)
                        
                        if iou > iou_threshold:
                            # High overlap detected
                            if det1['confidence'] <= det2['confidence']:
                                # Current detection has lower confidence, don't keep it
                                should_keep = False
                                logger.debug(f"Detectron2 IoU filter: Removing {class_name} "
                                           f"conf={det1['confidence']:.3f} (IoU={iou:.3f} with "
                                           f"conf={det2['confidence']:.3f})")
                                break
                            else:
                                # Current detection has higher confidence, remove the previous one
                                keep_indices.remove(j)
                                logger.debug(f"Detectron2 IoU filter: Replacing {class_name} "
                                           f"conf={det2['confidence']:.3f} with "
                                           f"conf={det1['confidence']:.3f} (IoU={iou:.3f})")
                
                if should_keep:
                    keep_indices.append(i)
            
            # Add the kept detections
            for i in keep_indices:
                filtered_detections.append(class_detections[i])
            
            logger.debug(f"Detectron2 IoU filter: {class_name} {len(class_detections)} → {len(keep_indices)} detections")
        
        return filtered_detections
    
    def is_healthy(self) -> bool:
        """Check if the analyzer is healthy and functional"""
        try:
            if self.demo is None:
                return False
            
            # Test actual functionality with a small test array
            test_array = np.zeros((100, 100, 3), dtype=np.uint8)
            self.analyze_from_array(test_array)
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def get_supported_classes(self) -> List[str]:
        """Get list of supported object classes"""
        return self.coco_classes.copy()
    
    def __del__(self):
        """Cleanup Detectron2 resources"""
        if hasattr(self, 'demo') and self.demo:
            # Detectron2 cleanup is handled internally
            pass