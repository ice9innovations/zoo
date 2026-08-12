#!/usr/bin/env python3
"""
Two-Stage CLIP Detector

Combines YOLO object detection with CLIP classification
for comprehensive object detection and identification.
"""

import logging
import time
import sys
import os
import requests
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image

from yolo import YOLODetector

logger = logging.getLogger(__name__)


class TwoStageCLIPDetector:
    """Two-stage object detector: YOLO11n detection + CLIP classification"""
    
    def __init__(self,
                 clip_service_url: str,
                 yolo_model_path: str = "yolo11n.pt", 
                 detection_threshold: float = 0.5,
                 max_detections: int = 30,
                 min_object_size: int = 20):
        """Initialize two-stage detector"""
        
        self.detection_threshold = detection_threshold
        self.clip_service_url = clip_service_url.rstrip('/')
        self.min_object_size = min_object_size
        
        # Initialize YOLO11n detector only (no local CLIP)
        self.detector = YOLODetector(
            model_path=yolo_model_path,
            confidence_threshold=detection_threshold,
            max_detections=max_detections
        )
        
        logger.info(f"✅ TwoStageCLIPDetector initialized with external CLIP at {clip_service_url}")
    
    def initialize(self) -> bool:
        """Initialize detection model and check CLIP service"""
        try:
            logger.info("Initializing detection model and checking CLIP service...")
            
            # Initialize detector
            if not self.detector.initialize():
                logger.error("Failed to initialize YOLO11n detector")
                return False
            
            # Check if CLIP service is available
            try:
                response = requests.get(f"{self.clip_service_url}/health", timeout=5)
                if response.status_code == 200:
                    logger.info(f"✅ CLIP service available at {self.clip_service_url}")
                else:
                    logger.error(f"CLIP service returned status {response.status_code}")
                    return False
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to connect to CLIP service: {e}")
                return False
            
            logger.info("✅ Detector initialized and CLIP service connected")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get combined model information"""
        return {
            "detection_model": self.detector.get_model_info(),
            "classification_service": f"{self.clip_service_url}",
            "architecture": "Two-stage: YOLO11n + External CLIP"
        }
    
    def analyze_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Perform two-stage analysis: detection + classification
        
        Args:
            image: PIL Image object
            
        Returns:
            Dict containing complete analysis results
        """
        start_time = time.time()
        
        try:
            # Stage 1: Object Detection
            detection_result = self.detector.detect_objects(image)
            
            if not detection_result["success"]:
                return detection_result
            
            detections = detection_result["data"]["detections"]
            
            # Filter out objects that are too small to classify reliably
            filtered_detections = []
            for detection in detections:
                bbox = detection["bbox"]
                width = bbox["width"]
                height = bbox["height"]
                
                # Skip objects smaller than minimum size
                if width < self.min_object_size or height < self.min_object_size:
                    logger.debug(f"Skipping small object {detection.get('id')}: {width}x{height} < {self.min_object_size}")
                    continue
                    
                filtered_detections.append(detection)
            
            logger.info(f"Filtered {len(detections)} detections to {len(filtered_detections)} (removed {len(detections) - len(filtered_detections)} too small)")
            
            # Stage 2: Classification on crops via HTTP (parallel)
            crops = self.detector.crop_objects(image, filtered_detections)
            classified_objects = []
            full_image_classification = None
            
            classification_start = time.time()
            
            # Process all crops in parallel + full image
            with ThreadPoolExecutor(max_workers=10) as executor:
                # Submit full image classification first
                future_to_detection = {}
                full_image_future = executor.submit(self._classify_via_http, image)
                future_to_detection[full_image_future] = {
                    "bbox": {"x": 0, "y": 0, "width": image.width, "height": image.height},
                    "is_full_image": True,
                    "id": "full_image"
                }
                
                # Submit all crop classification tasks
                for cropped_image, detection in crops:
                    future = executor.submit(self._classify_via_http, cropped_image)
                    future_to_detection[future] = detection
                
                # Collect results as they complete
                for future in as_completed(future_to_detection):
                    detection = future_to_detection[future]
                    try:
                        classification_result = future.result()
                        
                        if classification_result and classification_result.get("predictions"):
                            predictions = classification_result["predictions"]
                            if predictions:
                                # Check if this is the full image classification
                                if detection.get("is_full_image"):
                                    # Keep ALL predictions for full image
                                    for prediction in predictions:
                                        # Normalize label (lowercase, spaces to underscores)
                                        raw_label = prediction.get("label", "unknown")
                                        normalized_label = raw_label.lower().strip().replace(' ', '_') if raw_label else "unknown"
                                        
                                        full_image_pred = {
                                            "label": normalized_label,
                                            "emoji": prediction.get("emoji", ""),
                                            "confidence": prediction.get("confidence", 0.0),
                                            "bbox": detection["bbox"],
                                            "is_full_image": True
                                        }
                                        classified_objects.append(full_image_pred)
                                else:
                                    # Regular object detection - keep only top prediction
                                    top_prediction = predictions[0]
                                    
                                    # Normalize label (lowercase, spaces to underscores)
                                    raw_label = top_prediction.get("label", "unknown")
                                    normalized_label = raw_label.lower().strip().replace(' ', '_') if raw_label else "unknown"
                                    
                                    classified_object = {
                                        "label": normalized_label,
                                        "emoji": top_prediction.get("emoji", ""),
                                        "confidence": top_prediction.get("confidence", 0.0),
                                        "bbox": detection["bbox"]
                                    }
                                    classified_objects.append(classified_object)
                    except Exception as e:
                        logger.debug(f"Failed to classify object {detection.get('id')}: {e}")
                        continue
            
            classification_time = time.time() - classification_start
            total_time = time.time() - start_time
            
            # Deduplicate same-label overlapping detections
            classified_objects = self._deduplicate_detections(classified_objects)
            
            # Sort by confidence (classification confidence)
            classified_objects.sort(key=lambda x: x["confidence"], reverse=True)
            
            # Separate full image predictions from object detections
            full_image_predictions = [obj for obj in classified_objects if obj.get("is_full_image")]
            object_predictions = [obj for obj in classified_objects if not obj.get("is_full_image")]
            
            # Sort both by confidence
            object_predictions.sort(key=lambda x: x["confidence"], reverse=True)
            full_image_predictions.sort(key=lambda x: x["confidence"], reverse=True)
            
            # Clean up full image predictions (remove is_full_image flag and bbox)
            cleaned_full_image = []
            for pred in full_image_predictions:
                clean_pred = pred.copy()
                clean_pred.pop("is_full_image", None)  # Remove the flag
                clean_pred.pop("bbox", None)  # Remove redundant bbox for full image
                cleaned_full_image.append(clean_pred)
            
            logger.info(f"Two-stage analysis: {len(detections)} detected, {len(filtered_detections)} filtered, {len(object_predictions)} objects with bboxes, {len(cleaned_full_image)} full image predictions")
            
            return {
                "success": True,
                "data": {
                    "predictions": object_predictions,
                    "full_image": cleaned_full_image,
                    "model_info": {
                        "framework": "YOLO + CLIP"
                    }
                },
                "processing_time": total_time
            }
            
        except Exception as e:
            logger.error(f"Two-stage analysis error: {str(e)}")
            return {
                "success": False,
                "error": f"Analysis failed: {str(e)}",
                "processing_time": time.time() - start_time
            }
    
    def _classify_via_http(self, image: Image.Image) -> Dict[str, Any]:
        """Send image to CLIP service for classification"""
        try:
            # Convert PIL Image to bytes
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            image_bytes = buffered.getvalue()
            
            # Send as multipart form data
            files = {'file': ('crop.jpg', image_bytes, 'image/jpeg')}
            response = requests.post(
                f"{self.clip_service_url}/analyze",
                files=files,
                timeout=2
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("predictions"):
                    logger.debug(f"Got {len(result['predictions'])} predictions from CLIP")
                return result
            else:
                logger.debug(f"CLIP service returned {response.status_code}")
                return None
                
        except Exception as e:
            logger.debug(f"Failed to classify crop: {e}")
            return None
    
    def _deduplicate_detections(self, detections):
        """Deduplicate same-label overlapping detections using bbox merger logic"""
        if not detections:
            return detections
        
        # Filter out full image classifications from deduplication
        regular_detections = [d for d in detections if not d.get("is_full_image")]
        full_image_detections = [d for d in detections if d.get("is_full_image")]
        
        if not regular_detections:
            return detections
        
        # Group by label
        label_groups = {}
        for detection in regular_detections:
            label = detection["label"]
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(detection)
        
        deduplicated = []
        
        for label, group_detections in label_groups.items():
            if len(group_detections) == 1:
                # Single detection, keep as-is
                deduplicated.extend(group_detections)
            else:
                # Multiple detections with same label, merge overlapping ones
                merged = self._merge_overlapping_detections(group_detections)
                deduplicated.extend(merged)
        
        # Add back any full image detections (don't deduplicate these)
        deduplicated.extend(full_image_detections)
        
        return deduplicated
    
    def _merge_overlapping_detections(self, detections):
        """Merge overlapping detections with same label"""
        clusters = []
        used = set()
        
        for i, detection in enumerate(detections):
            if i in used:
                continue
            
            cluster = [detection]
            used.add(i)
            
            # Find overlapping detections
            for j in range(i + 1, len(detections)):
                if j in used:
                    continue
                
                overlap = self._calculate_overlap_ratio(detection["bbox"], detections[j]["bbox"])
                if overlap > 0.3:  # 30% overlap threshold
                    cluster.append(detections[j])
                    used.add(j)
            
            clusters.append(cluster)
        
        # Merge each cluster
        merged_detections = []
        for cluster in clusters:
            if len(cluster) == 1:
                merged_detections.append(cluster[0])
            else:
                merged = self._merge_cluster(cluster)
                merged_detections.append(merged)
        
        return merged_detections
    
    def _merge_cluster(self, cluster):
        """Merge a cluster of overlapping detections"""
        # Keep highest confidence detection as base
        best_detection = max(cluster, key=lambda d: d["confidence"])
        
        # Calculate merged bounding box
        boxes = [d["bbox"] for d in cluster]
        x1 = min(b["x"] for b in boxes)
        y1 = min(b["y"] for b in boxes)
        x2 = max(b["x"] + b["width"] for b in boxes)
        y2 = max(b["y"] + b["height"] for b in boxes)
        
        merged_bbox = {
            "x": x1,
            "y": y1,
            "width": x2 - x1,
            "height": y2 - y1
        }
        
        # Create merged detection
        return {
            "label": best_detection["label"],
            "confidence": best_detection["confidence"],  # Keep best confidence
            "bbox": merged_bbox,
            "emoji": best_detection.get("emoji", ""),
            "merged_count": len(cluster)
        }
    
    def _calculate_overlap_ratio(self, box1, box2):
        """Calculate IoU overlap ratio"""
        x1 = max(box1["x"], box2["x"])
        y1 = max(box1["y"], box2["y"])
        x2 = min(box1["x"] + box1["width"], box2["x"] + box2["width"])
        y2 = min(box1["y"] + box1["height"], box2["y"] + box2["height"])
        
        if x1 >= x2 or y1 >= y2:
            return 0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = box1["width"] * box1["height"]
        area2 = box2["width"] * box2["height"]
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
