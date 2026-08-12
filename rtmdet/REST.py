#!/usr/bin/env python3
"""
RTMDet Object Detection REST API Service
Provides high-performance object detection using RTMDet model.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchvision.ops
import json
import requests
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse
from PIL import Image, ImageDraw
import numpy as np
import time
import random

from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Configuration for GitHub raw file downloads (optional - fallback to local config)
TIMEOUT = float(os.getenv('TIMEOUT', '10.0'))  # Default 10 seconds for GitHub requests
AUTO_UPDATE = os.getenv('AUTO_UPDATE', 'True').lower() == 'true'  # Enable/disable GitHub downloads

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Local analyzer module
from rtmdet_analyzer import RTMDetAnalyzer

try:
    from mmdet.apis import init_detector, inference_detector
    import mmcv
    MMDET_AVAILABLE = True
    logger.info("MMDetection available")
except ImportError as e:
    logger.warning(f"MMDetection not available: {e}")
    MMDET_AVAILABLE = False

# Load emoji mappings from local JSON file
emoji_mappings = {}

def load_emoji_mappings():
    """Load emoji mappings from GitHub raw files with local caching"""
    global emoji_mappings
    local_cache_path = os.path.join(os.path.dirname(__file__), 'emoji_mappings.json')

    # Try GitHub raw file first if AUTO_UPDATE is enabled
    if AUTO_UPDATE:
        github_url = "https://raw.githubusercontent.com/ice9innovations/animal-farm/refs/heads/main/config/emoji_mappings.json"

        try:
            logger.info(f"🔄 RTMDet: Loading fresh emoji mappings from GitHub: {github_url}")
            response = requests.get(github_url, timeout=TIMEOUT)
            response.raise_for_status()
            data = response.json()

            # Cache to disk for future offline use
            try:
                with open(local_cache_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                logger.info(f"💾 RTMDet: Cached emoji mappings to {local_cache_path}")
            except Exception as cache_error:
                logger.warning(f"⚠️  RTMDet: Failed to cache emoji mappings: {cache_error}")

            emoji_mappings = data
            logger.info("✅ RTMDet: Successfully loaded emoji mappings from GitHub")
            return
        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️  RTMDet: Failed to load emoji mappings from GitHub: {e}")
            logger.info("🔄 RTMDet: Falling back to local cache due to GitHub failure")
    else:
        logger.info("🔄 RTMDet: AUTO_UPDATE disabled, using local cache only")

    # Fallback to local cached file
    try:
        logger.info(f"🔄 RTMDet: Loading emoji mappings from local cache: {local_cache_path}")
        with open(local_cache_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        emoji_mappings = data
        logger.info("✅ RTMDet: Successfully loaded emoji mappings from local cache")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"❌ RTMDet: Failed to load local emoji mappings from {local_cache_path}: {e}")
        if AUTO_UPDATE:
            raise Exception(f"Failed to load emoji mappings from both GitHub and local cache: {e}")
        else:
            raise Exception(f"Failed to load emoji mappings - AUTO_UPDATE disabled and no local cache available. Set AUTO_UPDATE=True or provide emoji_mappings.json in rtmdet directory: {e}")

def get_emoji(word: str) -> str:
    """Simple emoji lookup - lowercase and replace spaces with underscores"""
    if not word:
        return None

    # Basic normalization: lowercase and replace spaces with underscores
    clean_word = word.lower().strip().replace(' ', '_')

    # Try exact match
    if clean_word in emoji_mappings:
        return emoji_mappings[clean_word]

    # Try singular form for common plurals
    if clean_word.endswith('s') and len(clean_word) > 3:
        singular = clean_word[:-1]
        if singular in emoji_mappings:
            return emoji_mappings[singular]

    return None

def check_shiny():
    """Check if this detection should be shiny (1/2500 chance)"""
    roll = random.randint(1, 2500)
    is_shiny = roll == 1
    return is_shiny, roll

def apply_nms_consolidation(detections: List[Dict[str, Any]], nms_threshold: float = 0.5, confidence_threshold: float = 0.25) -> List[Dict[str, Any]]:
    """
    Apply Non-Maximum Suppression to eliminate overlapping detections.
    This is the core function that actually removes overlapping bounding boxes.
    """
    if not detections:
        return []

    # Filter by confidence threshold first
    valid_detections = [det for det in detections if det.get('confidence', 0) >= confidence_threshold]

    if len(valid_detections) <= 1:
        return valid_detections

    # Convert detections to tensors for NMS
    boxes = []
    scores = []

    for detection in valid_detections:
        bbox = detection.get('bbox', {})
        if not bbox:
            continue

        # Convert from RTMDet format {x1, y1, width, height} to PyTorch format [x1, y1, x2, y2]
        x1 = float(bbox.get('x1', 0))
        y1 = float(bbox.get('y1', 0))
        width = float(bbox.get('width', 0))
        height = float(bbox.get('height', 0))
        x2 = x1 + width
        y2 = y1 + height

        boxes.append([x1, y1, x2, y2])
        scores.append(float(detection.get('confidence', 0)))

    if not boxes:
        return []

    # Convert to PyTorch tensors
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)

    # Apply NMS - this is where overlapping detections are actually eliminated
    keep_indices = torchvision.ops.nms(boxes_tensor, scores_tensor, nms_threshold)

    # Return only the detections that survived NMS
    consolidated_detections = []
    for idx in keep_indices:
        consolidated_detections.append(valid_detections[idx.item()])

    # Sort by confidence (highest first)
    consolidated_detections.sort(key=lambda x: x.get('confidence', 0), reverse=True)

    return consolidated_detections

# Configuration
MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', str(32 * 1024 * 1024)))  # 32MB default
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
RAW_IMAGE_CONTENT_TYPES = {
    'image/jpeg',
    'image/png',
    'image/webp',
    'application/octet-stream',
}
PRIVATE_STR = os.getenv('PRIVATE')
PORT_STR = os.getenv('PORT')
CONFIDENCE_THRESHOLD_STR = os.getenv('CONFIDENCE_THRESHOLD')

# Validate critical configuration
if not PRIVATE_STR:
    raise ValueError("PRIVATE environment variable is required")
if not PORT_STR:
    raise ValueError("PORT environment variable is required")

# Convert to appropriate types
PRIVATE = PRIVATE_STR.lower() == 'true'
PORT = int(PORT_STR)
CONFIDENCE_THRESHOLD = float(CONFIDENCE_THRESHOLD_STR) if CONFIDENCE_THRESHOLD_STR else 0.25
IOU_THRESHOLD = 0.45  # IoU threshold for NMS
MAX_DETECTIONS = 100  # Maximum number of detections per image

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.backends.mps.is_available():
    device = 'mps'

logger.info(f"Using device: {device}")

# Global variables
analyzer = None
coco_classes = []

# COCO class names (RTMDet uses COCO dataset classes)
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

def load_coco_classes() -> bool:
    """Load COCO class names"""
    global coco_classes
    try:
        coco_classes = COCO_CLASSES
        logger.info(f"Loaded {len(coco_classes)} COCO classes")
        return True
    except Exception as e:
        logger.error(f"Failed to load COCO classes: {e}")
        return False

def initialize_rtmdet_analyzer() -> bool:
    """Initialize RTMDet analyzer once at startup"""
    global analyzer
    try:
        logger.info("Initializing RTMDet Analyzer...")

        analyzer = RTMDetAnalyzer(
            confidence_threshold=CONFIDENCE_THRESHOLD,
            coco_classes=coco_classes,
            device=device
        )

        logger.info("✅ RTMDet Analyzer initialized successfully")
        return True

    except Exception as e:
        logger.error(f"❌ Error initializing RTMDet Analyzer: {str(e)}")
        return False

def is_allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def is_raw_image_request() -> bool:
    return (request.content_type or '').split(';', 1)[0].strip().lower() in RAW_IMAGE_CONTENT_TYPES


def lookup_emoji(class_name: str) -> Optional[str]:
    """Look up emoji for a given class name using local emoji service (optimized - no HTTP requests)"""
    clean_name = class_name.lower().strip()

    try:
        # Use simple local emoji lookup
        emoji = get_emoji(clean_name)
        if emoji:
            logger.debug(f"Local emoji lookup: '{clean_name}' → {emoji}")
            return emoji

        logger.debug(f"Local emoji lookup: no emoji found for '{clean_name}'")
        return None

    except Exception as e:
        logger.warning(f"Local emoji service lookup failed for '{clean_name}': {e}")
        return None

def process_image_for_detection(image: Image.Image) -> Dict[str, Any]:
    """
    Main processing function - takes PIL Image, returns detection data
    This orchestrates the analyzer call - pure business logic orchestration
    """
    try:
        if not analyzer:
            return {
                "success": False,
                "error": "RTMDet analyzer not loaded"
            }

        # Use analyzer for core ML processing
        result = analyzer.analyze_from_pil_image(image)

        # Apply NMS consolidation to eliminate overlapping detections
        detections = result.get('detections', [])
        if detections:
            detections = apply_nms_consolidation(detections)
            result['detections'] = detections

        # Add emoji information to detections
        for detection in detections:
            class_name = detection.get('class_name', '')
            try:
                emoji = lookup_emoji(class_name)
                if emoji:
                    detection['emoji'] = emoji
            except Exception as e:
                logger.warning(f"Emoji lookup failed for '{class_name}': {e}")

        return {
            "success": True,
            "data": result
        }

    except Exception as e:
        logger.error(f"Error in image processing orchestration: {e}")
        return {
            "success": False,
            "error": f"Detection failed: {str(e)}"
        }

def create_rtmdet_response(data: Dict[str, Any], processing_time: float = None) -> Dict[str, Any]:
    """Create standardized rtmdet response"""
    detections = data.get("detections", [])

    # Use processing time from data if not provided
    if processing_time is None:
        processing_time = data.get('processing_time', 0)

    # Create unified prediction format
    predictions = []
    for detection in detections:
        bbox = detection.get('bbox', {})
        is_shiny, shiny_roll = check_shiny()

        prediction = {
            "label": detection.get('class_name', ''),
            "confidence": round(float(detection.get('confidence', 0)), 3)
        }

        # Add shiny flag only for shiny detections
        if is_shiny:
            prediction["shiny"] = True
            logger.info(f"✨ SHINY {detection.get('class_name', '').upper()} DETECTED! Roll: {shiny_roll} ✨")

        # Add bbox if present (convert from RTMDet format)
        if bbox:
            prediction["bbox"] = {
                "x": bbox.get('x1', 0),
                "y": bbox.get('y1', 0),
                "width": bbox.get('width', 0),
                "height": bbox.get('height', 0)
            }

        # Add emoji if present
        if detection.get('emoji'):
            prediction["emoji"] = detection['emoji']

        predictions.append(prediction)

    # Sort predictions by confidence (highest first)
    predictions.sort(key=lambda x: x['confidence'], reverse=True)

    return {
        "service": "rtmdet",
        "status": "success",
        "predictions": predictions,
        "metadata": {
            "processing_time": round(processing_time, 3),
            "model_info": {
                "framework": "MMDetection RTMDet"
            }
        }
    }

# Initialize Flask app
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

# Enable CORS for direct browser access (eliminates PHP proxy)
CORS(app, origins=["*"], methods=["GET", "POST", "OPTIONS"])
print("RTMDet service: CORS enabled for direct browser communication")

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
    """Health check endpoint with fail-fast validation"""
    global analyzer
    try:
        # Check if analyzer is initialized and functional
        if analyzer is None:
            return jsonify({
                'status': 'unhealthy',
                'service': 'rtmdet',
                'error': 'RTMDet analyzer not initialized'
            }), 503

        # Test actual functionality
        if not analyzer.is_healthy():
            return jsonify({
                'status': 'unhealthy',
                'service': 'rtmdet',
                'error': 'RTMDet analyzer non-functional'
            }), 503

        return jsonify({
            'status': 'healthy',
            'service': 'rtmdet',
            'capabilities': ['object_detection', 'bbox_extraction'],
            'models': {
                'object_detection': {
                    'status': 'ready',
                    'framework': 'MMDetection RTMDet',
                    'confidence_threshold': CONFIDENCE_THRESHOLD,
                    'classes_loaded': len(COCO_CLASSES)
                }
            },
            'supported_classes': len(COCO_CLASSES),
            'endpoints': [
                "GET /health - Health check",
                "GET /analyze?url=<image_url> - Analyze objects from URL",
                "GET /analyze?file=<file_path> - Analyze objects from file",
                "POST /analyze - Analyze objects from uploaded file",
                "GET /v3/analyze?url=<image_url> - V3 compatibility",
                "GET /v2/analyze?image_url=<image_url> - V2 compatibility"
            ],
            'timestamp': time.time()
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'service': 'rtmdet',
            'error': str(e)
        }), 500

@app.route('/classes', methods=['GET'])
def get_classes():
    """Get supported object classes"""
    return jsonify({
        "classes": COCO_CLASSES,
        "total_classes": len(COCO_CLASSES),
        "framework": "MMDetection RTMDet"
    })

@app.route('/', methods=['GET'])
def analyze_get():
    """GET endpoint for frontend compatibility - uses in-memory processing"""
    image_url = request.args.get('url')
    if not image_url:
        # Return health check if no URL provided
        if analyzer is not None:
            return jsonify({
                "status": "healthy",
                "service": "rtmdet",
                "version": "1.0.0",
                "model_loaded": True
            })
        else:
            return jsonify({
                "status": "error",
                "service": "rtmdet",
                "version": "1.0.0",
                "model_loaded": False,
                "error": "RTMDet analyzer not available"
            }), 503

    start_time = time.time()

    try:
        # Download image directly into memory
        parsed_url = urlparse(image_url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("Invalid URL format")

        response = requests.get(image_url, timeout=10)
        response.raise_for_status()

        content_type = response.headers.get('content-type', '')
        if not content_type.startswith('image/'):
            raise ValueError("URL does not point to an image")

        # Process image in memory
        from io import BytesIO
        image = Image.open(BytesIO(response.content))

        # Process using in-memory detection
        processing_result = process_image_for_detection(image)

        if not processing_result.get('success'):
            return jsonify({
                "status": "error",
                "error": processing_result.get('error', 'Detection failed'),
                "processing_time": round(time.time() - start_time, 3)
            }), 500

        # Get detection data
        result_data = processing_result.get('data', {})
        detections = result_data.get('detections', [])

        # Create unified prediction format for legacy compatibility
        predictions = []
        for detection in detections:
            bbox = detection.get('bbox', {})
            prediction = {
                "type": "object_detection",
                "label": detection.get('class_name', ''),
                "confidence": float(detection.get('confidence', 0)),
                "bbox": {
                    "x": bbox.get('x1', 0),
                    "y": bbox.get('y1', 0),
                    "width": bbox.get('width', 0),
                    "height": bbox.get('height', 0)
                }
            }

            # Add emoji if present
            if detection.get('emoji'):
                prediction["emoji"] = detection['emoji']

            predictions.append(prediction)

        return jsonify({
            "service": "rtmdet",
            "status": "success",
            "predictions": predictions,
            "metadata": {
                "processing_time": round(time.time() - start_time, 3),
                "model_info": {
                    "name": "RTMDet",
                    "framework": "MMDetection"
                },
                "image_dimensions": result_data.get('image_dimensions', {})
            }
        })

    except Exception as e:
        logger.error(f"Error in legacy GET endpoint: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "processing_time": round(time.time() - start_time, 3)
        }), 500

@app.route('/v2/analyze', methods=['POST'])
def analyze_v2():
    """V2 API endpoint with unified response format - uses in-memory processing"""
    start_time = time.time()

    try:
        # Get request data
        data = request.get_json()
        if not data or 'image_url' not in data:
            return jsonify({
                "service": "rtmdet",
                "status": "error",
                "predictions": [],
                "error": {"message": "Missing image_url in request"},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 400

        image_url = data['image_url']

        # Download image directly into memory
        parsed_url = urlparse(image_url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("Invalid URL format")

        response = requests.get(image_url, timeout=10)
        response.raise_for_status()

        content_type = response.headers.get('content-type', '')
        if not content_type.startswith('image/'):
            raise ValueError("URL does not point to an image")

        # Process image in memory
        from io import BytesIO
        image = Image.open(BytesIO(response.content))

        # Process using in-memory detection
        processing_result = process_image_for_detection(image)

        if not processing_result.get('success'):
            return jsonify({
                "service": "rtmdet",
                "status": "error",
                "predictions": [],
                "error": {"message": processing_result.get('error', 'Detection failed')},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 500

        # Get detection data
        result_data = processing_result.get('data', {})
        detections = result_data.get('detections', [])

        # Create unified prediction format for v2 compatibility
        predictions = []
        for detection in detections:
            bbox = detection.get('bbox', {})
            prediction = {
                "type": "object_detection",
                "label": detection.get('class_name', ''),
                "confidence": float(detection.get('confidence', 0)),
                "bbox": {
                    "x": bbox.get('x1', 0),
                    "y": bbox.get('y1', 0),
                    "width": bbox.get('width', 0),
                    "height": bbox.get('height', 0)
                }
            }

            # Add emoji if present
            if detection.get('emoji'):
                prediction["emoji"] = detection['emoji']

            predictions.append(prediction)

        return jsonify({
            "service": "rtmdet",
            "status": "success",
            "predictions": predictions,
            "metadata": {
                "processing_time": round(time.time() - start_time, 3),
                "model_info": {
                    "name": "RTMDet",
                    "framework": "MMDetection",
                    "confidence_threshold": CONFIDENCE_THRESHOLD,
                    "device": device
                },
                "image_dimensions": result_data.get('image_dimensions', {})
            }
        })

    except Exception as e:
        logger.error(f"Error in v2 analyze endpoint: {e}")
        return jsonify({
            "service": "rtmdet",
            "status": "error",
            "predictions": [],
            "error": {"message": str(e)},
            "metadata": {"processing_time": round(time.time() - start_time, 3)}
        }), 500

@app.route('/v3/analyze', methods=['GET', 'POST'])
def analyze_v3():
    """Unified V3 analyze endpoint - orchestrates input handling and processing"""
    start_time = time.time()

    def error_response(message: str, status_code: int = 400):
        return jsonify({
            "service": "rtmdet",
            "status": "error",
            "predictions": [],
            "error": {"message": message},
            "metadata": {"processing_time": round(time.time() - start_time, 3)}
        }), status_code

    try:
        # Step 1: Get image into memory from any source
        if request.method == 'POST' and is_raw_image_request():
            try:
                from io import BytesIO
                file_data = request.get_data(cache=False)
                if not file_data:
                    return error_response("No image body provided")
                if len(file_data) > MAX_FILE_SIZE:
                    return error_response(f"File too large. Maximum size: {MAX_FILE_SIZE//1024//1024}MB")
                image = Image.open(BytesIO(file_data)).convert('RGB')
            except Exception as e:
                return error_response(f"Failed to process raw image body: {str(e)}", 500)

        elif request.method == 'POST' and 'file' in request.files:
            # Handle file upload
            uploaded_file = request.files['file']
            if uploaded_file.filename == '':
                return error_response("No file selected")

            # Validate file size
            uploaded_file.seek(0, 2)  # Seek to end
            file_size = uploaded_file.tell()
            uploaded_file.seek(0)     # Seek back to beginning

            if file_size > MAX_FILE_SIZE:
                return error_response(f"File too large. Maximum size: {MAX_FILE_SIZE//1024//1024}MB")

            # Validate file type
            if not is_allowed_file(uploaded_file.filename):
                return error_response("File type not allowed")

            try:
                from io import BytesIO
                file_data = uploaded_file.read()
                image = Image.open(BytesIO(file_data)).convert('RGB')
            except Exception as e:
                return error_response(f"Failed to process uploaded image: {str(e)}", 500)

        else:
            # Handle URL or file parameter
            url = request.args.get('url')
            file_path = request.args.get('file')

            if not url and not file_path:
                return error_response("Must provide either 'url' or 'file' parameter, or POST a file")

            if url and file_path:
                return error_response("Cannot provide both 'url' and 'file' parameters")

            if url:
                # Download from URL directly into memory
                try:
                    from urllib.parse import urlparse
                    parsed_url = urlparse(url)
                    if not parsed_url.scheme or not parsed_url.netloc:
                        return error_response("Invalid URL format")

                    response = requests.get(url, timeout=10, stream=True)
                    response.raise_for_status()

                    content_type = response.headers.get('content-type', '')
                    if not content_type.startswith('image/'):
                        return error_response("URL does not point to an image")

                    # Check content length if provided
                    content_length = response.headers.get('content-length')
                    if content_length and int(content_length) > MAX_FILE_SIZE:
                        return error_response("Downloaded file too large")

                    from io import BytesIO
                    image_data = BytesIO()
                    for chunk in response.iter_content(chunk_size=8192):
                        image_data.write(chunk)
                        if image_data.tell() > MAX_FILE_SIZE:
                            return error_response("Downloaded file too large")

                    image_data.seek(0)
                    image = Image.open(image_data).convert('RGB')

                except Exception as e:
                    return error_response(f"Failed to download/process image: {str(e)}")
            else:  # file_path
                # Load file directly into memory
                if not os.path.exists(file_path):
                    return error_response(f"File not found: {file_path}")

                if not is_allowed_file(file_path):
                    return error_response("File type not allowed")

                if not validate_file_size(file_path):
                    return error_response("File too large")

                try:
                    image = Image.open(file_path).convert('RGB')
                except Exception as e:
                    return error_response(f"Failed to load image file: {str(e)}", 500)

        # Step 2: Process the image (unified processing path)
        processing_result = process_image_for_detection(image)

        # Step 3: Handle processing result
        if not processing_result["success"]:
            return error_response(processing_result["error"], 500)

        # Step 4: Create response
        response = create_rtmdet_response(processing_result["data"])

        return jsonify(response)

    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        logger.error(f"V3 Analyze API error: {e}")
        return error_response(f"Internal error: {str(e)}", 500)

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    """Main analyze endpoint - calls v3 analyze directly"""
    try:
        return analyze_v3()
    except Exception as e:
        logger.error(f"Analyze API error: {e}")
        return jsonify({
            "service": "rtmdet",
            "status": "error",
            "predictions": [],
            "error": {"message": f"Endpoint error: {str(e)}"},
            "metadata": {"processing_time": 0}
        }), 500

if __name__ == '__main__':
    # Initialize services
    logger.info("Starting RTMDet service...")

    # Load emoji mappings on startup
    load_emoji_mappings()

    coco_loaded = load_coco_classes()
    model_loaded = initialize_rtmdet_analyzer()

    if not model_loaded:
        logger.error("Failed to load RTMDet analyzer.")
        logger.error("Please ensure MMDetection is installed and RTMDet models are available.")
        logger.error("Service cannot function without analyzer. Exiting.")
        exit(1)

    if not coco_loaded:
        logger.error("Failed to load COCO classes.")
        logger.error("Service cannot function without class definitions. Exiting.")
        exit(1)

    # Always use 0.0.0.0 to allow external connections
    host = "0.0.0.0"

    logger.info(f"Starting RTMDet service on {host}:{PORT}")
    logger.info(f"Private mode: {PRIVATE}")
    logger.info(f"Model loaded: {model_loaded}")
    logger.info(f"COCO classes loaded: {coco_loaded}")
    logger.info("Emoji lookup: Local file mode")
    logger.info(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")

    app.run(
        host=host,
        port=PORT,
        debug=False,
        use_reloader=False,
        threaded=True
    )
