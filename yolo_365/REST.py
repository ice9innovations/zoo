#!/usr/bin/env python3
"""
YOLOv11 Object Detection REST API Service
Provides object detection using Ultralytics YOLOv11 model with custom Object365 training.
"""

import json
import requests
import os
import logging
import random
import time
import torch
import torchvision.ops
from typing import List, Dict, Any
from urllib.parse import urlparse
from PIL import Image
from io import BytesIO

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from yolo_analyzer import YoloAnalyzer

# Load environment variables first
load_dotenv()

# Configuration for GitHub raw file downloads (optional - fallback to local config)
TIMEOUT = float(os.getenv('TIMEOUT', '10.0'))  # Default 10 seconds for GitHub requests
AUTO_UPDATE = os.getenv('AUTO_UPDATE', 'True').lower() == 'true'  # Enable/disable GitHub downloads

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables as strings first
AUTO_UPDATE_STR = os.getenv('AUTO_UPDATE', 'true')
PORT_STR = os.getenv('PORT')
PRIVATE_STR = os.getenv('PRIVATE')
CONFIDENCE_THRESHOLD_STR = os.getenv('CONFIDENCE_THRESHOLD')
DATASET = os.getenv('DATASET')
MODEL_PATH = os.getenv('MODEL_PATH')
SERVICE_NAME = os.getenv('SERVICE_NAME', 'YOLO')

# Validate critical environment variables
if not PORT_STR:
    raise ValueError("PORT environment variable is required")
if not PRIVATE_STR:
    raise ValueError("PRIVATE environment variable is required")

# Convert to appropriate types after validation
AUTO_UPDATE = AUTO_UPDATE_STR.lower() == 'true'
PORT = int(PORT_STR)
PRIVATE = PRIVATE_STR.lower() in ['true', '1', 'yes']
CONFIDENCE_THRESHOLD = float(CONFIDENCE_THRESHOLD_STR) if CONFIDENCE_THRESHOLD_STR else 0.25

# Configuration
MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', str(32 * 1024 * 1024)))  # 32MB default
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
RAW_IMAGE_CONTENT_TYPES = {
    'image/jpeg',
    'image/png',
    'image/webp',
    'application/octet-stream',
}
IOU_THRESHOLD = 0.3  # IoU threshold for NMS
MAX_DETECTIONS = 100  # Maximum number of detections per image

# Global analyzer instance
yolo_analyzer = None


def is_raw_image_request() -> bool:
    return (request.content_type or '').split(';', 1)[0].strip().lower() in RAW_IMAGE_CONTENT_TYPES

def load_emoji_mappings():
    """Load emoji mappings from GitHub raw files with local caching"""
    local_cache_path = os.path.join(os.path.dirname(__file__), 'emoji_mappings.json')

    # Try GitHub raw file first if AUTO_UPDATE is enabled
    if AUTO_UPDATE:
        github_url = "https://raw.githubusercontent.com/ice9innovations/animal-farm/refs/heads/main/config/emoji_mappings.json"

        try:
            logger.info(f"🔄 {SERVICE_NAME}: Loading fresh emoji mappings from GitHub: {github_url}")
            response = requests.get(github_url, timeout=TIMEOUT)
            response.raise_for_status()
            data = response.json()

            # Cache to disk for future offline use
            try:
                with open(local_cache_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                logger.info(f"💾 {SERVICE_NAME}: Cached emoji mappings to {local_cache_path}")
            except Exception as cache_error:
                logger.warning(f"⚠️  {SERVICE_NAME}: Failed to cache emoji mappings: {cache_error}")

            logger.info(f"✅ {SERVICE_NAME}: Successfully loaded emoji mappings from GitHub")
            return data
        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️  {SERVICE_NAME}: Failed to load emoji mappings from GitHub: {e}")
            logger.info(f"🔄 {SERVICE_NAME}: Falling back to local cache due to GitHub failure")
    else:
        logger.info(f"🔄 {SERVICE_NAME}: AUTO_UPDATE disabled, using local cache only")

    # Fallback to local cached file
    try:
        logger.info(f"🔄 {SERVICE_NAME}: Loading emoji mappings from local cache: {local_cache_path}")
        with open(local_cache_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"✅ {SERVICE_NAME}: Successfully loaded emoji mappings from local cache")
        return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"❌ {SERVICE_NAME}: Failed to load local emoji mappings from {local_cache_path}: {e}")
        if AUTO_UPDATE:
            raise Exception(f"Failed to load emoji mappings from both GitHub and local cache: {e}")
        else:
            raise Exception(f"Failed to load emoji mappings - AUTO_UPDATE disabled and no local cache available. Set AUTO_UPDATE=True or provide emoji_mappings.json in service directory: {e}")

def check_shiny() -> tuple[bool, int]:
    """Check if this detection should be shiny (1/2500 chance)"""
    roll = random.randint(1, 2500)
    is_shiny = roll == 1
    return is_shiny, roll

def is_allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_yolo11_response(data: Dict[str, Any], processing_time: float) -> Dict[str, Any]:
    """Create standardized YOLOv11 response with object detections"""
    detections = data.get('detections', [])
    
    # Create unified prediction format
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
        
        # Add shiny flag only for shiny detections
        if is_shiny:
            prediction["shiny"] = True
            logger.info(f"✨ SHINY {detection.get('class_name', '').upper()} DETECTED! Roll: {shiny_roll} ✨")
        
        # Add emoji if present
        if detection.get('emoji'):
            prediction["emoji"] = detection['emoji']
        
        predictions.append(prediction)

    # Sort predictions by confidence (highest first)
    predictions.sort(key=lambda x: x['confidence'], reverse=True)

    return {
        "service": SERVICE_NAME if SERVICE_NAME else "yolo",
        "status": "success",
        "predictions": predictions,
        "metadata": {
            "processing_time": round(processing_time, 3),
            "model_info": yolo_analyzer.get_model_info() if yolo_analyzer else {}
        }
    }

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

        # YOLO-365 format should be {x, y, width, height} - convert to [x1, y1, x2, y2] for NMS
        x = float(bbox.get('x', 0))
        y = float(bbox.get('y', 0))
        width = float(bbox.get('width', 0))
        height = float(bbox.get('height', 0))
        x2 = x + width
        y2 = y + height

        boxes.append([x, y, x2, y2])
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

def download_image_from_url(url: str) -> Image.Image:
    """Download image from URL and return as PIL Image (in-memory processing)"""
    try:
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("Invalid URL format")
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        content_type = response.headers.get('content-type', '')
        if not content_type.startswith('image/'):
            raise ValueError("URL does not point to an image")
        
        if len(response.content) > MAX_FILE_SIZE:
            raise ValueError("Downloaded file too large")
        
        # Return PIL Image directly from bytes (no temp files)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        return image
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to download image: {str(e)}")

def validate_image_file(file_path: str) -> Image.Image:
    """Validate and load image file as PIL Image (in-memory processing)"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if not is_allowed_file(file_path):
        raise ValueError("File type not allowed")
    
    try:
        # Load directly into memory, no temp files
        image = Image.open(file_path).convert('RGB')
        return image
    except Exception as e:
        raise Exception(f"Failed to load image: {str(e)}")

def initialize_yolo_analyzer() -> bool:
    """Initialize YOLO analyzer once at startup - fail fast"""
    global yolo_analyzer
    try:
        logger.info("Initializing YOLO Analyzer...")

        # Load emoji mappings first
        try:
            emoji_mappings = load_emoji_mappings()
        except Exception as e:
            logger.warning(f"⚠️  {SERVICE_NAME}: Failed to load emoji mappings: {e}")
            emoji_mappings = {}

        yolo_analyzer = YoloAnalyzer(
            model_path=MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            iou_threshold=IOU_THRESHOLD,
            max_detections=MAX_DETECTIONS,
            service_name=SERVICE_NAME,
            dataset=DATASET,
            emoji_mappings=emoji_mappings
        )

        # Initialize the model
        if not yolo_analyzer.initialize():
            logger.error("❌ Failed to initialize YOLO Analyzer")
            return False

        logger.info("✅ YOLO Analyzer initialized successfully")
        return True

    except Exception as e:
        logger.error(f"❌ Error initializing YOLO Analyzer: {str(e)}")
        return False

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

# Enable CORS for direct browser access (eliminates PHP proxy)
CORS(app, origins=["*"], methods=["GET", "POST", "OPTIONS"])
print("YOLOv11 service: CORS enabled for direct browser communication")

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
    # Test if YOLOv11 analyzer is actually working
    try:
        if not yolo_analyzer:
            raise ValueError("Analyzer not loaded")
        
        # Test with a small dummy image
        test_image = Image.new('RGB', (100, 100), color='blue')
        test_result = yolo_analyzer.analyze_from_array(test_image)
        
        if not test_result.get('success'):
            raise ValueError(f"Analyzer test failed: {test_result.get('error')}")
        
        analyzer_status = "loaded"
        status = "healthy"
        
    except Exception as e:
        analyzer_status = f"error: {str(e)}"
        status = "unhealthy"
        
        return jsonify({
            "status": status,
            "reason": f"YOLOv11 analyzer error: {str(e)}",
            "service": "YOLOv11 Object Detection"
        }), 503
    
    model_info = yolo_analyzer.get_model_info() if yolo_analyzer else {}
    
    return jsonify({
        "status": status,
        "service": "YOLOv11 Object Detection",
        "analyzer": {
            "status": analyzer_status,
            **model_info
        },
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "custom_model": bool(MODEL_PATH and os.path.exists(MODEL_PATH)),
        "endpoints": [
            "GET /health - Health check",
            "GET,POST /analyze - Unified endpoint (URL/file/upload)",
            "GET /v3/analyze - V3 compatibility",
            "GET /classes - Get supported classes",
            "GET /v2/analyze - V2 compatibility (deprecated)",
            "GET /v2/analyze_file - V2 compatibility (deprecated)"
        ]
    })

@app.route('/classes', methods=['GET'])
def get_classes():
    """Get supported object classes"""
    if yolo_analyzer:
        try:
            class_names = yolo_analyzer.get_supported_classes()
            return jsonify({
                "classes": class_names,
                "total_classes": len(class_names),
                "source": "Custom Model" if (MODEL_PATH and os.path.exists(MODEL_PATH)) else "Standard YOLO"
            })
        except Exception as e:
            logger.warning(f"Error getting class names from analyzer: {e}")
    
    return jsonify({
        "classes": [],
        "total_classes": 0,
        "source": "Analyzer not loaded or classes unavailable"
    })

# V2 Compatibility Routes - Translate parameters and call analyze
@app.route('/v2/analyze', methods=['GET'])
def analyze_v2_compat():
    """V2 compatibility - translate parameters to new analyze format"""
    image_url = request.args.get('image_url')
    
    if image_url:
        # Parameter translation: image_url -> url
        new_args = {'url': image_url}
        with app.test_request_context('/analyze', query_string=new_args):
            return analyze()
    else:
        # Let new analyze handle validation errors
        with app.test_request_context('/analyze'):
            return analyze()

@app.route('/v2/analyze_file', methods=['GET'])
def analyze_file_v2_compat():
    """V2 file compatibility - translate parameters to new analyze format"""
    file_path = request.args.get('file_path')
    
    if file_path:
        # Parameter translation: file_path -> file
        new_args = {'file': file_path}
        with app.test_request_context('/analyze', query_string=new_args):
            return analyze()
    else:
        with app.test_request_context('/analyze'):
            return analyze()

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    """Unified analyze endpoint - orchestrates input handling and processing (pure in-memory)"""
    start_time = time.time()
    
    def error_response(message: str, status_code: int = 400):
        return jsonify({
            "service": SERVICE_NAME if SERVICE_NAME else "yolo",
            "status": "error",
            "predictions": [],
            "error": {"message": message},
            "metadata": {"processing_time": round(time.time() - start_time, 3)}
        }), status_code
    
    try:
        # Step 1: Get image into memory from any source (NO FILE SYSTEM OPERATIONS)
        if request.method == 'POST' and is_raw_image_request():
            try:
                file_data = request.get_data(cache=False)
                if not file_data:
                    return error_response("No image body provided")
                if len(file_data) > MAX_FILE_SIZE:
                    return error_response(f"File too large. Maximum size: {MAX_FILE_SIZE//1024//1024}MB")
                image = Image.open(BytesIO(file_data)).convert('RGB')
            except Exception as e:
                return error_response(f"Failed to process raw image body: {str(e)}", 500)

        elif request.method == 'POST' and 'file' in request.files:
            # Handle file upload - pure in-memory processing
            uploaded_file = request.files['file']
            if uploaded_file.filename == '':
                return error_response("No file selected")
            
            if not is_allowed_file(uploaded_file.filename):
                return error_response("File type not allowed")
            
            # Validate file size
            uploaded_file.seek(0, 2)  # Seek to end
            file_size = uploaded_file.tell()
            uploaded_file.seek(0)     # Seek back to beginning
            
            if file_size > MAX_FILE_SIZE:
                return error_response(f"File too large. Maximum size: {MAX_FILE_SIZE//1024//1024}MB")
            
            try:
                # Pure in-memory processing - no temp files
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
                    image = download_image_from_url(url)
                except Exception as e:
                    return error_response(f"Failed to download/process image: {str(e)}")
            else:  # file_path
                # Load file directly into memory
                try:
                    image = validate_image_file(file_path)
                except Exception as e:
                    return error_response(f"Failed to load image file: {str(e)}", 500)
        
        # Step 2: Process the image using the analyzer (unified processing path)
        if not yolo_analyzer:
            return error_response("YOLO analyzer not initialized", 500)
        
        processing_result = yolo_analyzer.analyze_from_array(image)
        
        # Step 3: Handle processing result
        if not processing_result["success"]:
            return error_response(processing_result["error"], 500)

        # Step 3.5: Apply NMS consolidation to eliminate overlapping detections
        detections = processing_result.get('detections', [])
        if detections:
            detections = apply_nms_consolidation(detections)
            processing_result['detections'] = detections

        # Step 4: Create response
        response = create_yolo11_response(
            processing_result,
            processing_result["processing_time"]
        )
        
        return jsonify(response)
        
    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        return error_response(f"Internal error: {str(e)}", 500)

@app.route('/v3/analyze', methods=['GET', 'POST'])
def analyze_v3_compat():
    """V3 compatibility - calls analyze function directly"""
    return analyze()


if __name__ == '__main__':
    # Initialize analyzer
    logger.info("Starting YOLOv11 service...")
    logger.info(f"Looking for model at: {MODEL_PATH}")
    
    analyzer_loaded = initialize_yolo_analyzer()
    
    if not analyzer_loaded:
        logger.error("Failed to load YOLOv11 analyzer. Service will run but detection will fail.")
        logger.error("Please ensure YOLOv11 models are available or install ultralytics: pip install ultralytics")
    
    # Determine host based on private mode
    host = "127.0.0.1" if PRIVATE else "0.0.0.0"
    
    logger.info(f"Starting YOLOv11 service on {host}:{PORT}")
    logger.info(f"Private mode: {PRIVATE}")
    logger.info(f"Analyzer loaded: {analyzer_loaded}")
    logger.info(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    logger.info(f"Custom model: {bool(MODEL_PATH and os.path.exists(MODEL_PATH))}")
    logger.info("🚀 In-memory processing enabled - no temp files created")
    
    app.run(
        host=host,
        port=PORT,
        debug=False,
        use_reloader=False,
        threaded=True
    )
