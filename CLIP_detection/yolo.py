#!/usr/bin/env python3
"""
YOLO Object Detection Stage

Handles YOLO model loading and object detection
using Ultralytics YOLO for fast inference with proper object bounding boxes.
"""

import os
# Set CUDA launch blocking to prevent segmentation faults - MUST be before any CUDA imports
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class YOLODetector:
    """YOLO object detection with proper bounding boxes"""
    
    def __init__(self, 
                 model_path: str = "yolov8x-seg.pt",
                 confidence_threshold: float = 0.5,
                 iou_threshold: float = 0.7,
                 max_detections: int = 30):
        """Initialize YOLO detector"""
        
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.model = None
        
        logger.info(f"✅ YOLODetector initialized - Model: {model_path}, Conf: {confidence_threshold}")
    
    def initialize(self) -> bool:
        """Initialize YOLO model"""
        try:
            logger.info(f"Loading YOLO model: {self.model_path}")
            self.model = YOLO(self.model_path)
            logger.info("✅ YOLO model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load YOLO model: {e}")
            return False
    
    def detect_objects(self, image: Image.Image) -> Dict[str, Any]:
        """
        Detect objects in image using YOLO
        
        Args:
            image: PIL Image object
            
        Returns:
            Dict containing detection results with bounding boxes
        """
        start_time = time.time()
        
        if not self.model:
            return {
                "success": False,
                "error": "YOLO model not loaded"
            }
        
        try:
            # Run YOLO detection
            detection_start = time.time()
            results = self.model(
                image, 
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                max_det=self.max_detections,
                verbose=False
            )
            detection_time = time.time() - detection_start
            
            # Process results
            detections = self._process_results(results[0], image.size)
            
            logger.info(f"Found {len(detections)} objects in {detection_time:.3f}s")
            
            return {
                "success": True,
                "data": {
                    "detections": detections,
                    "total_detections": len(detections),
                    "image_dimensions": {
                        "width": image.size[0],
                        "height": image.size[1]
                    },
                    "model_info": {
                        "confidence_threshold": self.confidence_threshold,
                        "detection_time": round(detection_time, 3),
                        "framework": "Ultralytics",
                        "model": "YOLO"
                    }
                },
                "processing_time": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Detection processing error: {str(e)}")
            return {
                "success": False,
                "error": f"Detection failed: {str(e)}",
                "processing_time": time.time() - start_time
            }
    
    def _process_results(self, result, image_size: Tuple[int, int]) -> List[Dict[str, Any]]:
        """Process FastSAM results into standardized format"""
        detections = []
        image_width, image_height = image_size
        
        if result.boxes is None:
            return detections
        
        boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
        confidences = result.boxes.conf.cpu().numpy()
        
        for i, (box, conf) in enumerate(zip(boxes, confidences)):
            x1, y1, x2, y2 = box.astype(int)
            
            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, image_width))
            y1 = max(0, min(y1, image_height))
            x2 = max(0, min(x2, image_width))
            y2 = max(0, min(y2, image_height))
            
            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue
            
            detection = {
                "id": i,
                "bbox": {
                    "x": int(x1),
                    "y": int(y1), 
                    "width": int(x2 - x1),
                    "height": int(y2 - y1)
                },
                "detection_confidence": round(float(conf), 3),
                "area": int((x2 - x1) * (y2 - y1))
            }
            
            detections.append(detection)
        
        return detections
    
    def crop_objects(self, image: Image.Image, detections: List[Dict[str, Any]]) -> List[Tuple[Image.Image, Dict[str, Any]]]:
        """
        Crop detected objects from image for classification
        
        Args:
            image: Original PIL Image
            detections: List of detection results
            
        Returns:
            List of (cropped_image, detection_info) tuples
        """
        crops = []
        
        for detection in detections:
            try:
                bbox = detection["bbox"]
                x1 = bbox["x"]
                y1 = bbox["y"]
                x2 = x1 + bbox["width"]
                y2 = y1 + bbox["height"]
                
                # Crop image
                cropped = image.crop((x1, y1, x2, y2))
                crops.append((cropped, detection))
                
            except Exception as e:
                logger.warning(f"Failed to crop object {detection.get('id', 'unknown')}: {e}")
                continue
        
        return crops
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_loaded": self.model is not None,
            "model_path": self.model_path,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "framework": "Ultralytics", 
            "model": "YOLO"
        }
