#!/usr/bin/env python3
"""
RTMDet Object Detection Analyzer - Core detection functionality

Handles object detection using MMDetection's RTMDet framework with enhanced
features for the standalone detection service.
"""

import os
import logging
import time
import threading
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

try:
    from mmdet.apis import init_detector, inference_detector
    import mmcv
    MMDET_AVAILABLE = True
except ImportError as e:
    logger.warning(f"MMDetection not available: {e}")
    MMDET_AVAILABLE = False


class RTMDetAnalyzer:
    """Core RTMDet object detection functionality"""

    def __init__(self,
                 confidence_threshold: float = 0.25,
                 coco_classes: List[str] = None,
                 device: str = 'cuda'):
        """Initialize RTMDet analyzer"""

        self.confidence_threshold = confidence_threshold
        self.coco_classes = coco_classes or []
        self.device = device
        self.model = None

        # Thread lock for model inference
        self.model_lock = threading.Lock()

        # IoU threshold for filtering overlapping detections
        self.iou_threshold = 0.45
        self.max_detections = 100

        # Initialize the model
        self._initialize_model()

    def _initialize_model(self):
        """Initialize RTMDet model using MMDetection"""
        if not MMDET_AVAILABLE:
            logger.error("MMDetection not available - RTMDet analyzer will not function")
            self.model = None
            return False

        try:
            # Try different RTMDet variants with proper model zoo format
            config_variants = [
                ('rtmdet_l_8xb32-300e_coco', 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_l_8xb32-300e_coco/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth'),
                ('rtmdet_m_8xb32-300e_coco', 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_m_8xb32-300e_coco/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth'),
                ('rtmdet_s_8xb32-300e_coco', 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_s_8xb32-300e_coco/rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth'),
                ('rtmdet_tiny_8xb32-300e_coco', 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth')
            ]

            model_loaded = False
            for config_name, checkpoint_url in config_variants:
                try:
                    logger.info(f"Attempting to load RTMDet variant: {config_name}")

                    # Initialize model from model zoo with proper checkpoint
                    self.model = init_detector(f'{config_name}.py', checkpoint_url, device=self.device)

                    logger.info(f"Successfully loaded RTMDet variant: {config_name}")
                    model_loaded = True
                    break

                except Exception as e:
                    logger.warning(f"Failed to load {config_name}: {e}")
                    continue

            if not model_loaded:
                logger.error("Failed to load any RTMDet variant - analyzer will not function")
                self.model = None
                return False

            logger.info(f"RTMDet analyzer initialized successfully on {self.device}")
            return True

        except Exception as e:
            logger.error(f"Error initializing RTMDet analyzer: {e}")
            self.model = None
            return False

    def is_healthy(self) -> bool:
        """Check if analyzer is healthy and functional"""
        if not MMDET_AVAILABLE:
            return False
        return self.model is not None

    def analyze_from_file_path(self, image_path: str) -> Dict[str, Any]:
        """Analyze image from file path"""
        try:
            # Validate image path
            if not os.path.exists(image_path):
                return {
                    "status": "error",
                    "error": f"Image file not found: {image_path}"
                }

            with self.model_lock:
                start_time = time.time()

                # Load and validate image
                with Image.open(image_path) as img:
                    img_width, img_height = img.size

                # Check if we have a real model available
                if self.model is None or not MMDET_AVAILABLE:
                    return {
                        "status": "error",
                        "error": "RTMDet analyzer unavailable: MMDetection dependencies not properly installed"
                    }

                return self._analyze_inference_input(image_path, img_width, img_height, start_time)

        except Exception as e:
            logger.error(f"Error in RTMDet analysis: {e}")
            return {
                "status": "error",
                "error": f"Analysis failed: {str(e)}"
            }

    def analyze_from_pil_image(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze PIL Image object directly in memory."""
        try:
            with self.model_lock:
                start_time = time.time()

                if self.model is None or not MMDET_AVAILABLE:
                    return {
                        "status": "error",
                        "error": "RTMDet analyzer unavailable: MMDetection dependencies not properly installed"
                    }

                image_rgb = image.convert('RGB')
                img_width, img_height = image_rgb.size
                image_array = np.array(image_rgb)

                return self._analyze_inference_input(image_array, img_width, img_height, start_time)

        except Exception as e:
            logger.error(f"Error in RTMDet analysis: {e}")
            return {
                "status": "error",
                "error": f"Analysis failed: {str(e)}"
            }

    def _analyze_inference_input(self, image_input: Any, img_width: int, img_height: int, start_time: float) -> Dict[str, Any]:
        """Run RTMDet inference on a file path or in-memory image array."""
        result = inference_detector(self.model, image_input)

        logger.info(f"RTMDet inference result type: {type(result)}")
        logger.info(f"RTMDet inference result attributes: {dir(result) if hasattr(result, '__dict__') else 'No attributes'}")

        detections = []

        if hasattr(result, 'pred_instances'):
            pred_instances = result.pred_instances

            bboxes = pred_instances.bboxes.cpu().numpy()
            scores = pred_instances.scores.cpu().numpy()
            labels = pred_instances.labels.cpu().numpy()

            logger.info(f"RTMDet found {len(bboxes)} raw detections")
            if len(scores) > 0:
                logger.info(f"Confidence scores range: {scores.min():.3f} to {scores.max():.3f}")
                logger.info(f"Confidence threshold: {self.confidence_threshold}")
                above_threshold = (scores >= self.confidence_threshold).sum()
                logger.info(f"Detections above threshold: {above_threshold}")

            for i in range(len(bboxes)):
                confidence = float(scores[i])
                logger.debug(f"Detection {i}: confidence={confidence:.3f}, threshold={self.confidence_threshold}")
                if confidence < self.confidence_threshold:
                    continue

                class_idx = int(labels[i])
                if class_idx >= len(self.coco_classes):
                    continue

                class_name = self.coco_classes[class_idx]
                x1, y1, x2, y2 = bboxes[i]
                detections.append({
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": {
                        "x1": int(x1),
                        "y1": int(y1),
                        "width": int(x2 - x1),
                        "height": int(y2 - y1)
                    }
                })
        else:
            for class_idx, class_detections in enumerate(result):
                if len(class_detections) == 0 or class_idx >= len(self.coco_classes):
                    continue

                class_name = self.coco_classes[class_idx]
                for detection in class_detections:
                    confidence = float(detection[4])
                    if confidence < self.confidence_threshold:
                        continue

                    x1, y1, x2, y2 = detection[:4]
                    detections.append({
                        "class_name": class_name,
                        "confidence": confidence,
                        "bbox": {
                            "x1": int(x1),
                            "y1": int(y1),
                            "width": int(x2 - x1),
                            "height": int(y2 - y1)
                        }
                    })

        detections.sort(key=lambda x: x['confidence'], reverse=True)
        detections = detections[:self.max_detections]
        processing_time = time.time() - start_time

        return {
            "status": "success",
            "detections": detections,
            "total_detections": len(detections),
            "image_dimensions": {
                "width": img_width,
                "height": img_height
            },
            "model_info": {
                "name": "RTMDet",
                "framework": "MMDetection",
                "confidence_threshold": self.confidence_threshold,
                "device": self.device
            },
            "processing_time": processing_time
        }
