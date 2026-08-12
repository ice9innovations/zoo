#!/usr/bin/env python3
"""
HAILO-Accelerated YOLO REST API Service
High-performance object detection using HAILO-8L hardware acceleration.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import requests
import os
import sys
import logging
import random
import time
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse
from PIL import Image
from io import BytesIO
import numpy as np

from dotenv import load_dotenv

# HAILO imports
from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
                             InputVStreamParams, OutputVStreamParams, FormatType)

# Load environment variables first
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
AUTO_UPDATE_STR = os.getenv('AUTO_UPDATE', 'true')
PORT_STR = os.getenv('PORT')
PRIVATE_STR = os.getenv('PRIVATE')

if not PORT_STR:
    raise ValueError("PORT environment variable is required")
if not PRIVATE_STR:
    raise ValueError("PRIVATE environment variable is required")

AUTO_UPDATE = AUTO_UPDATE_STR.lower() == 'true'
PORT = int(PORT_STR)
PRIVATE = PRIVATE_STR.lower() in ['true', '1', 'yes']

# Configuration
MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', str(32 * 1024 * 1024)))  # 32MB default
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
RAW_IMAGE_CONTENT_TYPES = {
    'image/jpeg',
    'image/png',
    'image/webp',
    'application/octet-stream',
}
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.3
MAX_DETECTIONS = 100
HEF_PATH = '/usr/share/hailo-models/yolov8s_h8l.hef'
INPUT_SIZE = 640  # HAILO model input size

# HAILO device and model
vdevice = None
hailo_model = None
network_group = None
input_vstreams_params = None
output_vstreams_params = None

# COCO class names (HAILO YOLOv8 uses COCO dataset classes)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Load emoji mappings from GitHub
emoji_mappings = {}


def is_raw_image_request() -> bool:
    return (request.content_type or '').split(';', 1)[0].strip().lower() in RAW_IMAGE_CONTENT_TYPES

def load_emoji_mappings():
    """Load emoji mappings from GitHub, fall back to local cache"""
    global emoji_mappings

    github_url = "https://raw.githubusercontent.com/ice9innovations/animal-farm/refs/heads/main/config/emoji_mappings.json"
    local_cache_path = 'emoji_mappings.json'

    if AUTO_UPDATE:
        try:
            logger.info(f"🔄 HAILO YOLO: Loading emoji mappings from GitHub: {github_url}")
            response = requests.get(github_url, timeout=10.0)
            response.raise_for_status()

            with open(local_cache_path, 'w', encoding='utf-8') as f:
                json.dump(response.json(), f, indent=2, ensure_ascii=False)

            emoji_mappings = response.json()
            logger.info(f"✅ HAILO YOLO: Loaded emoji mappings from GitHub and cached locally ({len(emoji_mappings)} entries)")
            return
        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️  HAILO YOLO: Failed to load emoji mappings from GitHub ({e}), falling back to local cache")

    # Fall back to local cache
    try:
        with open(local_cache_path, 'r') as f:
            emoji_mappings = json.load(f)
            logger.info(f"✅ HAILO YOLO: Loaded emoji mappings from local cache ({len(emoji_mappings)} entries)")
    except Exception as local_error:
        logger.error(f"❌ HAILO YOLO: Failed to load local emoji mappings: {local_error}")
        raise Exception(f"Both GitHub and local emoji mappings failed: GitHub download disabled or failed, Local cache={local_error}")

# Load emoji mappings on startup
load_emoji_mappings()

def get_emoji(word: str) -> str:
    """Get emoji using direct mapping lookup"""
    if not word:
        return None
    word_clean = word.lower().strip()
    return emoji_mappings.get(word_clean)

def check_shiny():
    """Check if this detection should be shiny (1/2500 chance)"""
    roll = random.randint(1, 2500)
    is_shiny = roll == 1
    return is_shiny, roll

def calculate_iou(box1: Dict[str, float], box2: Dict[str, float]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes"""
    x1_1, y1_1 = box1['x'], box1['y']
    x2_1, y2_1 = x1_1 + box1['width'], y1_1 + box1['height']

    x1_2, y1_2 = box2['x'], box2['y']
    x2_2, y2_2 = x1_2 + box2['width'], y1_2 + box2['height']

    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)

    if x1_inter >= x2_inter or y1_inter >= y2_inter:
        return 0.0

    intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    area1 = box1['width'] * box1['height']
    area2 = box2['width'] * box2['height']
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0

def apply_iou_filtering(detections: List[Dict], iou_threshold: float = IOU_THRESHOLD) -> List[Dict]:
    """Apply IoU-based filtering to merge overlapping detections of the same class"""
    if not detections:
        return detections

    class_groups = {}
    for detection in detections:
        class_name = detection.get('class_name', '')
        if class_name not in class_groups:
            class_groups[class_name] = []
        class_groups[class_name].append(detection)

    filtered_detections = []

    for class_name, class_detections in class_groups.items():
        if len(class_detections) == 1:
            filtered_detections.extend(class_detections)
            continue

        keep_indices = []

        for i, det1 in enumerate(class_detections):
            should_keep = True

            for j in keep_indices:
                det2 = class_detections[j]
                bbox1 = det1.get('bbox', {})
                bbox2 = det2.get('bbox', {})

                if all(k in bbox1 for k in ['x', 'y', 'width', 'height']) and \
                   all(k in bbox2 for k in ['x', 'y', 'width', 'height']):
                    iou = calculate_iou(bbox1, bbox2)

                    if iou > iou_threshold:
                        if det1['confidence'] <= det2['confidence']:
                            should_keep = False
                            logger.debug(f"HAILO YOLO IoU filter: Removing {class_name} "
                                         f"conf={det1['confidence']:.3f} (IoU={iou:.3f} with "
                                         f"conf={det2['confidence']:.3f})")
                            break
                        else:
                            keep_indices.remove(j)
                            logger.debug(f"HAILO YOLO IoU filter: Replacing {class_name} "
                                         f"conf={det2['confidence']:.3f} with "
                                         f"conf={det1['confidence']:.3f} (IoU={iou:.3f})")

            if should_keep:
                keep_indices.append(i)

        for i in keep_indices:
            filtered_detections.append(class_detections[i])

        logger.debug(f"HAILO YOLO IoU filter: {class_name} {len(class_detections)} → {len(keep_indices)} detections")

    return filtered_detections

def initialize_hailo_model(model_path: str = None) -> bool:
    """Initialize HAILO model with persistent device and VStream parameters"""
    global hailo_model, vdevice, network_group, input_vstreams_params, output_vstreams_params

    try:
        model_file = model_path or HEF_PATH

        logger.info(f"Initializing HAILO-8L device...")
        vdevice = VDevice()

        logger.info(f"Loading HAILO model: {model_file}")
        hef = HEF(model_file)
        configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_groups = vdevice.configure(hef, configure_params)
        network_group = network_groups[0]

        input_vstreams_params = InputVStreamParams.make_from_network_group(
            network_group, quantized=True, format_type=FormatType.UINT8)
        output_vstreams_params = OutputVStreamParams.make_from_network_group(
            network_group, quantized=True, format_type=FormatType.FLOAT32)

        logger.info(f"HAILO YOLO model loaded successfully!")
        logger.info(f"Model device: HAILO-8L Hardware Acceleration")
        logger.info(f"Input layer: {list(input_vstreams_params.keys())}")
        logger.info(f"Output layers: {list(output_vstreams_params.keys())}")

        hailo_model = hef
        return True

    except Exception as e:
        logger.error(f"Failed to load HAILO model: {e}")
        return False

def lookup_emoji(class_name: str) -> Optional[str]:
    """Look up emoji for a given class name"""
    clean_name = class_name.lower().strip()
    try:
        emoji = get_emoji(clean_name)
        if emoji:
            logger.debug(f"Local emoji lookup: '{clean_name}' → {emoji}")
            return emoji
        logger.debug(f"Local emoji lookup: no emoji found for '{clean_name}'")
        return None
    except Exception as e:
        logger.warning(f"Local emoji service lookup failed for '{clean_name}': {e}")
        return None

def is_allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def letterbox_image(image: Image.Image) -> Tuple[np.ndarray, float, int, int]:
    """Resize image to INPUT_SIZE x INPUT_SIZE with letterboxing (aspect-ratio preserving).

    Pads with gray (128, 128, 128) so the model sees a neutral background rather than
    a stretched/squashed version of the image. Returns the preprocessed array and the
    letterbox parameters needed to map detected coordinates back to the original image.

    Returns:
        img_array: uint8 numpy array of shape (INPUT_SIZE, INPUT_SIZE, 3)
        scale:     scale factor applied to both dimensions
        pad_x:     horizontal padding in pixels (added to each side)
        pad_y:     vertical padding in pixels (added to each side)
    """
    orig_w, orig_h = image.size
    scale = min(INPUT_SIZE / orig_w, INPUT_SIZE / orig_h)
    new_w = round(orig_w * scale)
    new_h = round(orig_h * scale)

    resized = image.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new('RGB', (INPUT_SIZE, INPUT_SIZE), (128, 128, 128))
    pad_x = (INPUT_SIZE - new_w) // 2
    pad_y = (INPUT_SIZE - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))

    return np.array(canvas).astype(np.uint8), scale, pad_x, pad_y

def recover_coords(x1_lb: float, y1_lb: float, x2_lb: float, y2_lb: float,
                   scale: float, pad_x: int, pad_y: int,
                   orig_w: int, orig_h: int) -> Tuple[int, int, int, int]:
    """Map bounding box coordinates from letterboxed INPUT_SIZE space back to
    the original image coordinate space.

    Args:
        x1_lb, y1_lb, x2_lb, y2_lb: coordinates in letterboxed pixel space
        scale:  the scale factor used during letterboxing
        pad_x:  horizontal padding applied during letterboxing
        pad_y:  vertical padding applied during letterboxing
        orig_w, orig_h: original image dimensions

    Returns:
        (x1, y1, x2, y2) clamped to the original image bounds
    """
    x1 = max(0, round((x1_lb - pad_x) / scale))
    y1 = max(0, round((y1_lb - pad_y) / scale))
    x2 = min(orig_w, round((x2_lb - pad_x) / scale))
    y2 = min(orig_h, round((y2_lb - pad_y) / scale))
    return x1, y1, x2, y2

def process_hailo_results(hailo_output, scale: float, pad_x: int, pad_y: int,
                          orig_w: int, orig_h: int) -> List[Dict[str, Any]]:
    """Process HAILO model results into structured format.

    Coordinates are returned in the original image's pixel space.
    """
    detections = []

    try:
        logger.debug(f"HAILO output type: {type(hailo_output)}")

        if isinstance(hailo_output, list):
            logger.debug(f"HAILO output is list with {len(hailo_output)} elements")
            if len(hailo_output) == 0:
                logger.warning("HAILO output list is empty - likely inference failed")
                return detections

            for i, elem in enumerate(hailo_output):
                if hasattr(elem, '__len__') and len(elem) > 0:
                    logger.debug(f"HAILO output[{i}]: type={type(elem)}, shape={getattr(elem, 'shape', 'no shape')}")
                    logger.debug(f"HAILO output[{i}] length: {len(elem)}")

            # Process as class-wise detections (80 classes, each with N detections)
            if len(hailo_output) == 80:
                logger.debug("Processing 80-class HAILO output format")

                for class_idx, class_detections in enumerate(hailo_output):
                    if hasattr(class_detections, '__len__') and len(class_detections) > 0:
                        logger.debug(f"Processing {len(class_detections)} detections for class {class_idx}")

                        class_name = COCO_CLASSES[class_idx] if class_idx < len(COCO_CLASSES) else f"class_{class_idx}"

                        for detection_data in class_detections:
                            try:
                                # Hailo NMS post-processor outputs [y_min, x_min, y_max, x_max, confidence]
                                y1_norm, x1_norm, y2_norm, x2_norm, confidence = detection_data
                                logger.debug(f"Class {class_idx} ({class_name}): "
                                             f"norm=[{x1_norm:.3f}, {y1_norm:.3f}, {x2_norm:.3f}, {y2_norm:.3f}], "
                                             f"conf={confidence:.3f}")

                                if confidence >= CONFIDENCE_THRESHOLD and confidence <= 1.0:
                                    # Normalized (0-1) → letterboxed pixel space → original image space
                                    x1_lb = x1_norm * INPUT_SIZE
                                    y1_lb = y1_norm * INPUT_SIZE
                                    x2_lb = x2_norm * INPUT_SIZE
                                    y2_lb = y2_norm * INPUT_SIZE

                                    x1, y1, x2, y2 = recover_coords(
                                        x1_lb, y1_lb, x2_lb, y2_lb,
                                        scale, pad_x, pad_y, orig_w, orig_h
                                    )

                                    if x2 > x1 and y2 > y1:
                                        try:
                                            emoji = lookup_emoji(class_name)
                                        except RuntimeError as e:
                                            logger.error(f"Emoji lookup failed for '{class_name}': {e}")
                                            raise RuntimeError(f"Detection failed due to emoji lookup failure: {e}")

                                        detections.append({
                                            "class_id": class_idx,
                                            "class_name": class_name,
                                            "confidence": round(float(confidence), 3),
                                            "bbox": {
                                                "x": x1,
                                                "y": y1,
                                                "width": x2 - x1,
                                                "height": y2 - y1
                                            },
                                            "emoji": emoji
                                        })
                                        logger.debug(f"Added detection: {class_name} conf={confidence:.3f} "
                                                     f"bbox=[{x1},{y1},{x2-x1},{y2-y1}]")

                            except (ValueError, IndexError) as e:
                                logger.debug(f"Skipping malformed detection in class {class_idx}: {e}")
                                continue

                logger.info(f"Processed {len(detections)} detections from class-wise HAILO output")
                detections.sort(key=lambda x: x['confidence'], reverse=True)
                return apply_iou_filtering(detections)[:MAX_DETECTIONS]

            # Fallback: try first element if not 80-class format
            first_elem = hailo_output[0]
            if hasattr(first_elem, '__len__') and len(first_elem) == 0:
                logger.warning("HAILO output first element is empty - likely inference failed")
                return detections
            hailo_array = np.array(first_elem) if isinstance(first_elem, list) else first_elem
        elif hasattr(hailo_output, 'shape'):
            hailo_array = hailo_output
        else:
            logger.error(f"Unknown HAILO output format: {type(hailo_output)}")
            return detections

        if not isinstance(hailo_array, np.ndarray):
            hailo_array = np.array(hailo_array)

        if hailo_array.size == 0:
            logger.warning("HAILO array is empty - inference likely failed")
            return detections

        logger.debug(f"HAILO array shape: {hailo_array.shape}, dtype: {hailo_array.dtype}")

        if len(hailo_array.shape) == 1:
            array_size = hailo_array.shape[0]

            if array_size == 40080:
                num_boxes = 8016
                reshaped = hailo_array.reshape(num_boxes, 5)
                logger.debug(f"Reshaped to {num_boxes} boxes with 5 values each")

                for box_data in reshaped:
                    x1_norm, y1_norm, x2_norm, y2_norm, confidence = box_data

                    if confidence >= CONFIDENCE_THRESHOLD and confidence <= 1.0:
                        x1_lb = x1_norm * INPUT_SIZE
                        y1_lb = y1_norm * INPUT_SIZE
                        x2_lb = x2_norm * INPUT_SIZE
                        y2_lb = y2_norm * INPUT_SIZE

                        x1, y1, x2, y2 = recover_coords(
                            x1_lb, y1_lb, x2_lb, y2_lb,
                            scale, pad_x, pad_y, orig_w, orig_h
                        )

                        if x2 > x1 and y2 > y1:
                            class_name = "object"
                            try:
                                emoji = lookup_emoji(class_name)
                            except RuntimeError as e:
                                raise RuntimeError(f"Detection failed due to emoji lookup failure: {e}")

                            detections.append({
                                "class_id": 0,
                                "class_name": class_name,
                                "confidence": round(float(confidence), 3),
                                "bbox": {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1},
                                "emoji": emoji
                            })
            else:
                logger.warning(f"Unexpected HAILO output size: {array_size}, cannot parse detections")

        elif len(hailo_array.shape) == 2:
            num_detections, values_per_detection = hailo_array.shape
            logger.debug(f"Processing {num_detections} detections with {values_per_detection} values each")

            if values_per_detection >= 5:
                for detection_data in hailo_array:
                    x1_norm, y1_norm, x2_norm, y2_norm, confidence = detection_data[:5]
                    logger.debug(f"Detection: norm=[{x1_norm:.3f}, {y1_norm:.3f}, {x2_norm:.3f}, {y2_norm:.3f}], conf={confidence:.3f}")

                    if confidence >= CONFIDENCE_THRESHOLD and confidence <= 1.0:
                        x1_lb = x1_norm * INPUT_SIZE
                        y1_lb = y1_norm * INPUT_SIZE
                        x2_lb = x2_norm * INPUT_SIZE
                        y2_lb = y2_norm * INPUT_SIZE

                        x1, y1, x2, y2 = recover_coords(
                            x1_lb, y1_lb, x2_lb, y2_lb,
                            scale, pad_x, pad_y, orig_w, orig_h
                        )

                        if x2 > x1 and y2 > y1:
                            class_name = "object"
                            try:
                                emoji = lookup_emoji(class_name)
                            except RuntimeError as e:
                                raise RuntimeError(f"Detection failed due to emoji lookup failure: {e}")

                            detections.append({
                                "class_id": 0,
                                "class_name": class_name,
                                "confidence": round(float(confidence), 3),
                                "bbox": {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1},
                                "emoji": emoji
                            })
                            logger.debug(f"Added detection conf={confidence:.3f}")
            else:
                logger.warning(f"Unexpected values per detection: {values_per_detection}")

        else:
            logger.warning(f"Unexpected HAILO output shape: {hailo_array.shape}")

    except Exception as e:
        logger.error(f"Error processing HAILO results: {e}")
        logger.error(f"Output info: shape={getattr(hailo_output, 'shape', 'no shape')}, type={type(hailo_output)}")

    logger.info(f"Processed {len(detections)} raw detections from HAILO output")
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    return apply_iou_filtering(detections)[:MAX_DETECTIONS]

def process_image_for_yolo(image: Image.Image) -> Dict[str, Any]:
    """Main processing function — takes a PIL Image, returns YOLO detection data.

    Letterboxes to INPUT_SIZE x INPUT_SIZE (preserving aspect ratio), runs HAILO
    inference, then maps all bounding boxes back to the original image coordinate space.
    No disk I/O at any point.
    """
    start_time = time.time()

    try:
        if not network_group or not input_vstreams_params or not output_vstreams_params:
            raise ValueError("HAILO model not loaded")

        if image.mode != 'RGB':
            image = image.convert('RGB')

        orig_w, orig_h = image.size
        logger.debug(f"Running HAILO detection on PIL Image ({orig_w}x{orig_h})")

        # Letterbox to INPUT_SIZE x INPUT_SIZE, preserving aspect ratio
        img_array, scale, pad_x, pad_y = letterbox_image(image)
        logger.debug(f"Letterboxed: scale={scale:.4f}, pad=({pad_x},{pad_y})")

        input_name = list(input_vstreams_params.keys())[0]
        logger.debug(f"Input layer name: {input_name}")
        input_data = {input_name: np.expand_dims(img_array, axis=0)}

        with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
            network_group_params = network_group.create_params()
            with network_group.activate(network_group_params):
                logger.debug("Running HAILO inference...")
                result = infer_pipeline.infer(input_data)
                logger.debug("HAILO inference completed")

        output_key = list(result.keys())[0]
        detection_output = result[output_key]

        # Strip batch dimension if present
        if hasattr(detection_output, 'shape') and len(detection_output.shape) > 1 and detection_output.shape[0] == 1:
            detection_data = detection_output[0]
        elif isinstance(detection_output, list) and len(detection_output) == 1:
            detection_data = detection_output[0]
        else:
            detection_data = detection_output

        detections = process_hailo_results(detection_data, scale, pad_x, pad_y, orig_w, orig_h)

        processing_time = round(time.time() - start_time, 3)
        logger.info(f"HAILO inference completed in {processing_time}s — {len(detections)} objects detected")

        return {
            "success": True,
            "data": {
                "detections": detections,
                "total_detections": len(detections),
                "image_dimensions": {"width": orig_w, "height": orig_h},
                "model_info": {
                    "confidence_threshold": CONFIDENCE_THRESHOLD,
                    "iou_threshold": IOU_THRESHOLD,
                    "device": "HAILO-8L Hardware Acceleration"
                }
            },
            "processing_time": processing_time
        }

    except Exception as e:
        processing_time = round(time.time() - start_time, 3)
        logger.error(f"Error during HAILO detection: {e}")
        return {
            "success": False,
            "error": f"Detection failed: {str(e)}",
            "processing_time": processing_time
        }

def create_yolo_response(data: Dict[str, Any], processing_time: float) -> Dict[str, Any]:
    """Create standardized YOLO response with object detections"""
    detections = data.get('detections', [])

    predictions = []
    for detection in detections:
        bbox = detection.get('bbox', {})
        is_shiny, shiny_roll = check_shiny()

        prediction = {
            "label": detection.get('class_name', ''),
            "confidence": float(detection.get('confidence', 0)),
            "bbox": {
                "x": bbox.get('x', 0),
                "y": bbox.get('y', 0),
                "width": bbox.get('width', 0),
                "height": bbox.get('height', 0)
            }
        }

        if is_shiny:
            prediction["shiny"] = True
            logger.info(f"✨ SHINY {detection.get('class_name', '').upper()} DETECTED! Roll: {shiny_roll} ✨")

        if detection.get('emoji'):
            prediction["emoji"] = detection['emoji']

        predictions.append(prediction)

    predictions.sort(key=lambda x: x['confidence'], reverse=True)

    return {
        "service": "hailo_yolo",
        "status": "success",
        "predictions": predictions,
        "metadata": {
            "processing_time": round(processing_time, 3),
            "model_info": {
                "framework": "HAILO-8L Hardware Acceleration"
            }
        }
    }

# Flask app setup
app = Flask(__name__)


def _normalize_health_payload(payload):
    """Ensure every /health response exposes the common Animal Farm health shape."""
    if not isinstance(payload, dict):
        return payload

    payload.setdefault("status", "healthy")
    payload.setdefault("schema_version", "health.v1")
    if not payload.get("service"):
        payload["service"] = str(globals().get("SERVICE_NAME") or __file__.replace("\\", "/").rstrip("/").split("/")[-2])

    warnings = payload.get("warnings", [])
    if warnings is None:
        warnings = []
    elif isinstance(warnings, str):
        warnings = [warnings]
    payload["warnings"] = warnings

    dependencies = payload.get("dependencies") if isinstance(payload.get("dependencies"), dict) else {}
    for key in ("llama_server", "ollama", "extraction_engines"):
        if key in payload and key not in dependencies:
            dependencies[key] = payload[key]
    payload["dependencies"] = dependencies

    model_value = payload.get("model")
    if isinstance(model_value, dict):
        model = dict(model_value)
    elif model_value is not None:
        payload.setdefault("model_name", model_value)
        model = {"name": model_value}
    else:
        model = {}

    if "models" in payload and "components" not in model:
        model["components"] = payload["models"]
    for source_key in ("detector", "analyzer", "ocr_engine"):
        source_value = payload.get(source_key)
        if isinstance(source_value, dict):
            model.setdefault("status", source_value.get("status"))
            model.setdefault("details", source_value)
    if "model_status" in payload and "status" not in model:
        model["status"] = payload["model_status"]
    if "model_loaded" in payload and "status" not in model:
        model["status"] = "loaded" if payload["model_loaded"] else "not_loaded"
    if "backend_status" in payload and "status" not in model:
        model["status"] = payload["backend_status"]
    if "device" in payload and "device" not in model:
        model["device"] = payload["device"]
    if "framework" in payload and "framework" not in model:
        model["framework"] = payload["framework"]
    if "backend" in payload and "backend" not in model:
        model["backend"] = payload["backend"]
    for threshold_key in ("threshold", "confidence_threshold", "detection_threshold", "classification_threshold"):
        if threshold_key in payload and threshold_key not in model:
            model[threshold_key] = payload[threshold_key]
    payload["model"] = model

    payload.setdefault("endpoints", [])
    return payload


@app.after_request
def _normalize_health_response(response):
    if request.path != "/health" or not response.is_json:
        return response
    payload = response.get_json(silent=True)
    normalized = _normalize_health_payload(payload)
    if normalized is not payload:
        return response
    response.set_data(app.json.dumps(normalized))
    response.content_type = "application/json"
    return response

app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

CORS(app, origins=["*"], methods=["GET", "POST", "OPTIONS"])
print("HAILO YOLO service: CORS enabled for direct browser communication")

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large", "status": "error"}), 413

@app.errorhandler(400)
def bad_request(e):
    return jsonify({"error": "Bad request", "status": "error"}), 400

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error", "status": "error"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    model_status = "loaded" if hailo_model else "not_loaded"
    model_info = {}

    if hailo_model:
        try:
            model_info = {
                "device": "HAILO-8L Hardware Acceleration",
                "framework": "HAILO Platform"
            }
        except Exception:
            pass

    return jsonify({
        "status": "healthy",
        "model_status": model_status,
        "model_info": model_info,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "supported_classes": len(COCO_CLASSES)
    })

@app.route('/classes', methods=['GET'])
def get_classes():
    """Get supported object classes"""
    return jsonify({
        "classes": COCO_CLASSES,
        "total_classes": len(COCO_CLASSES)
    })

@app.route('/v2/analyze', methods=['GET'])
def analyze_v2_compat():
    """V2 compatibility - translate parameters to new analyze format"""
    image_url = request.args.get('image_url')
    if image_url:
        with app.test_request_context('/analyze', query_string={'url': image_url}):
            return analyze()
    else:
        with app.test_request_context('/analyze'):
            return analyze()

@app.route('/v2/analyze_file', methods=['GET'])
def analyze_file_v2_compat():
    """V2 file compatibility - translate parameters to new analyze format"""
    file_path = request.args.get('file_path')
    if file_path:
        with app.test_request_context('/analyze', query_string={'file': file_path}):
            return analyze()
    else:
        with app.test_request_context('/analyze'):
            return analyze()

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    """Unified analyze endpoint — fully in-memory, no disk I/O."""
    start_time = time.time()

    def error_response(message: str, status_code: int = 400):
        return jsonify({
            "service": "hailo_yolo",
            "status": "error",
            "predictions": [],
            "error": {"message": message},
            "metadata": {"processing_time": round(time.time() - start_time, 3)}
        }), status_code

    try:
        image = None

        if request.method == 'POST' and is_raw_image_request():
            try:
                file_data = request.get_data(cache=False)
                if not file_data:
                    return error_response("No image body provided")
                if len(file_data) > MAX_FILE_SIZE:
                    return error_response(f"File too large. Maximum size: {MAX_FILE_SIZE // 1024 // 1024}MB")
                image = Image.open(BytesIO(file_data)).convert('RGB')
            except Exception as e:
                return error_response(f"Failed to process raw image body: {str(e)}", 500)

        elif request.method == 'POST' and 'file' in request.files:
            uploaded_file = request.files['file']
            if uploaded_file.filename == '':
                return error_response("No file selected")

            uploaded_file.seek(0, 2)
            file_size = uploaded_file.tell()
            uploaded_file.seek(0)

            if file_size > MAX_FILE_SIZE:
                return error_response(f"File too large. Maximum size: {MAX_FILE_SIZE // 1024 // 1024}MB")

            try:
                file_data = uploaded_file.read()
                image = Image.open(BytesIO(file_data)).convert('RGB')
            except Exception as e:
                return error_response(f"Failed to process uploaded image: {str(e)}", 500)

        else:
            url = request.args.get('url')
            file = request.args.get('file')

            if not url and not file:
                return error_response("Must provide either 'url' or 'file' parameter, or POST a file")

            if url and file:
                return error_response("Cannot provide both 'url' and 'file' parameters")

            if url:
                try:
                    parsed_url = urlparse(url)
                    if not parsed_url.scheme or not parsed_url.netloc:
                        return error_response("Invalid URL format")

                    response = requests.get(url, timeout=10)
                    response.raise_for_status()

                    content_type = response.headers.get('content-type', '')
                    if not content_type.startswith('image/'):
                        return error_response("URL does not point to an image")

                    if len(response.content) > MAX_FILE_SIZE:
                        return error_response("Downloaded file too large")

                    image = Image.open(BytesIO(response.content)).convert('RGB')

                except Exception as e:
                    return error_response(f"Failed to download/process image: {str(e)}")

            else:  # file path
                if not os.path.exists(file):
                    return error_response(f"File not found: {file}", 404)

                if not is_allowed_file(file):
                    return error_response("File type not allowed")

                try:
                    image = Image.open(file).convert('RGB')
                except Exception as e:
                    return error_response(f"Failed to load image file: {str(e)}", 500)

        processing_result = process_image_for_yolo(image)

        if not processing_result["success"]:
            return error_response(processing_result["error"], 500)

        response = create_yolo_response(
            processing_result["data"],
            processing_result["processing_time"]
        )

        return jsonify(response)

    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        logger.error(f"Analyze error: {e}")
        return error_response(f"Internal error: {str(e)}", 500)

@app.route('/v3/analyze', methods=['GET', 'POST'])
def analyze_v3_compat():
    """V3 compatibility - calls analyze directly"""
    return analyze()

if __name__ == '__main__':
    logger.info("Starting HAILO-Accelerated YOLO service...")

    model_loaded = initialize_hailo_model()

    if not model_loaded:
        logger.error("Failed to load HAILO model. Please ensure HAILO drivers are installed "
                     f"and model file exists at: {HEF_PATH}")
        sys.exit(1)

    host = "127.0.0.1" if PRIVATE else "0.0.0.0"

    logger.info(f"Starting HAILO YOLO service on {host}:{PORT}")
    logger.info(f"Private mode: {PRIVATE}")
    logger.info(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    logger.info(f"Framework: HAILO-8L Hardware Acceleration")
    logger.info(f"Supported classes: {len(COCO_CLASSES)}")

    app.run(
        host=host,
        port=PORT,
        debug=False,
        use_reloader=False,
        threaded=True
    )
