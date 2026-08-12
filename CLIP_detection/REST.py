#!/usr/bin/env python3
"""
Two-Stage CLIP Detection REST API Service
Provides two-stage object detection: YOLO detection + CLIP classification.
"""

import os
# Simple GPU configuration for PyTorch/YOLO only
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Synchronous CUDA operations for debugging
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use only first GPU

import json
import requests
import os
import sys
import logging
import random
import time
import gc
from typing import List, Dict, Any
from urllib.parse import urlparse
from PIL import Image
from io import BytesIO

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from two_stage_clip_detector import TwoStageCLIPDetector

# Load environment variables first
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
PORT_STR = os.getenv('PORT')
PRIVATE_STR = os.getenv('PRIVATE')
DETECTION_THRESHOLD_STR = os.getenv('DETECTION_THRESHOLD')
CLASSIFICATION_THRESHOLD_STR = os.getenv('CLASSIFICATION_THRESHOLD')
MAX_DETECTIONS_STR = os.getenv('MAX_DETECTIONS')
SERVICE_NAME = os.getenv('SERVICE_NAME')
YOLO_MODEL_PATH = os.getenv('YOLO_MODEL_PATH')
CLIP_SERVICE_URL = os.getenv('CLIP_SERVICE_URL')

# Validate critical environment variables
if not PORT_STR:
    raise ValueError("PORT environment variable is required")
if not PRIVATE_STR:
    raise ValueError("PRIVATE environment variable is required")
if not SERVICE_NAME:
    raise ValueError("SERVICE_NAME environment variable is required")
if not YOLO_MODEL_PATH:
    raise ValueError("YOLO_MODEL_PATH environment variable is required")
if not CLIP_SERVICE_URL:
    raise ValueError("CLIP_SERVICE_URL environment variable is required")

# Convert to appropriate types after validation
PORT = int(PORT_STR)
PRIVATE = PRIVATE_STR.lower() in ['true', '1', 'yes']
DETECTION_THRESHOLD = float(DETECTION_THRESHOLD_STR) if DETECTION_THRESHOLD_STR else 0.5
CLASSIFICATION_THRESHOLD = float(CLASSIFICATION_THRESHOLD_STR) if CLASSIFICATION_THRESHOLD_STR else 0.15
MAX_DETECTIONS = int(MAX_DETECTIONS_STR) if MAX_DETECTIONS_STR else 30

# Configuration
MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', str(32 * 1024 * 1024)))  # 32MB default
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
RAW_IMAGE_CONTENT_TYPES = {
    'image/jpeg',
    'image/png',
    'image/webp',
    'application/octet-stream',
}

# Global detector instance
detector = None

def check_shiny() -> tuple[bool, int]:
    """Check if this detection should be shiny (1/2500 chance)"""
    roll = random.randint(1, 2500)
    is_shiny = roll == 1
    return is_shiny, roll

def is_allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def is_raw_image_request() -> bool:
    return (request.content_type or '').split(';', 1)[0].strip().lower() in RAW_IMAGE_CONTENT_TYPES


def create_detector_response(data: Dict[str, Any], processing_time: float) -> Dict[str, Any]:
    """Create standardized response with object detections"""
    predictions = data.get('predictions', [])
    full_image = data.get('full_image', [])
    
    # Add shiny chance to predictions
    for prediction in predictions:
        is_shiny, shiny_roll = check_shiny()
        if is_shiny:
            prediction["shiny"] = True
            logger.info(f"✨ SHINY {prediction.get('label', '').upper()} DETECTED! Roll: {shiny_roll} ✨")
    
    # Add shiny chance to full_image predictions
    for prediction in full_image:
        is_shiny, shiny_roll = check_shiny()
        if is_shiny:
            prediction["shiny"] = True
            logger.info(f"✨ SHINY {prediction.get('label', '').upper()} DETECTED! Roll: {shiny_roll} ✨")
    
    response = {
        "service": SERVICE_NAME,
        "status": "success",
        "predictions": predictions,
        "metadata": {
            "processing_time": round(processing_time, 3),
            "model_info": data.get('model_info', {})
        }
    }
    
    # Only include full_image if it exists and has content
    if full_image:
        response["full_image"] = full_image
    
    return response

def download_image_from_url(url: str) -> Image.Image:
    """Download image from URL and return as PIL Image (in-memory processing)"""
    try:
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("Invalid URL format")
        
        response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
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

def initialize_detector() -> bool:
    """Initialize the two-stage CLIP detector once at startup - fail fast"""
    global detector
    try:
        logger.info("Initializing two-stage CLIP detector...")
        
        detector = TwoStageCLIPDetector(
            clip_service_url=CLIP_SERVICE_URL,
            yolo_model_path=YOLO_MODEL_PATH,
            detection_threshold=DETECTION_THRESHOLD,
            max_detections=MAX_DETECTIONS
        )
        
        # Initialize the models
        if not detector.initialize():
            logger.error("❌ Failed to initialize detector")
            return False
            
        logger.info("✅ Detector initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error initializing detector: {str(e)}")
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

# Enable CORS for direct browser access (using flask-cors 4.0.0)
CORS(app, origins=["*"], methods=["GET", "POST", "OPTIONS"])
print(f"{SERVICE_NAME}: CORS enabled for direct browser communication")

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
    # Test if detector is actually working
    try:
        if not detector:
            raise ValueError("Detector not loaded")
        
        # Test with a small dummy image
        test_image = Image.new('RGB', (100, 100), color='blue')
        test_result = detector.analyze_image(test_image)
        
        if not test_result.get('success'):
            raise ValueError(f"Detector test failed: {test_result.get('error')}")
        
        detector_status = "loaded"
        status = "healthy"
        
    except Exception as e:
        detector_status = f"error: {str(e)}"
        status = "unhealthy"
        
        return jsonify({
            "status": status,
            "reason": f"Detector error: {str(e)}",
            "service": SERVICE_NAME
        }), 503
    
    model_info = detector.get_model_info() if detector else {}
    
    return jsonify({
        "status": status,
        "service": SERVICE_NAME,
        "detector": {
            "status": detector_status,
            **model_info
        },
        "detection_threshold": DETECTION_THRESHOLD,
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "max_detections": MAX_DETECTIONS,
        "endpoints": [
            "GET /health - Health check",
            "GET,POST /analyze - Unified endpoint (URL/file/upload)",
            "GET /classes - Get supported classes"
        ]
    })

@app.route('/classes', methods=['GET'])
def get_classes():
    """Get supported classes"""
    # For now, return basic info since classification model uses ImageNet classes
    return jsonify({
        "classes": "ImageNet 1000 classes",
        "total_classes": 1000,
        "source": "Classification model (ImageNet)",
        "note": "Full class list available via model documentation"
    })

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    """Unified analyze endpoint - orchestrates input handling and processing (pure in-memory)"""
    start_time = time.time()
    
    def error_response(message: str, status_code: int = 400):
        return jsonify({
            "service": SERVICE_NAME,
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
        
        # Step 2: Process the image using the detector (unified processing path)
        if not detector:
            return error_response("Detector not initialized", 500)
        
        processing_result = detector.analyze_image(image)
        
        # Step 3: Handle processing result
        if not processing_result["success"]:
            return error_response(processing_result["error"], 500)
        
        # Step 4: Create response
        response = create_detector_response(
            processing_result["data"],
            processing_result["processing_time"]
        )
        
        return jsonify(response)
        
    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        return error_response(f"Internal error: {str(e)}", 500)

@app.route('/v3/analyze', methods=['GET', 'POST'])
def analyze_v3_compat():
    """V3 compatibility - redirect to new analyze endpoint"""
    if request.method == 'POST':
        # Forward POST request with data
        return analyze()
    else:
        # Forward GET request with query string
        with app.test_request_context('/analyze', query_string=request.args):
            return analyze()


if __name__ == '__main__':
    # Initialize detector
    logger.info(f"Starting {SERVICE_NAME} service...")
    logger.info(f"Looking for model at: {YOLO_MODEL_PATH}")
    
    detector_loaded = initialize_detector()
    
    if not detector_loaded:
        logger.error("❌ CRITICAL: Failed to load detector. Exiting...")
        logger.error("Please ensure detection and classification models are available")
        sys.exit(1)  # Exit if detector fails to load
    
    # Determine host based on private mode
    host = "127.0.0.1" if PRIVATE else "0.0.0.0"
    
    logger.info(f"Starting {SERVICE_NAME} service on {host}:{PORT}")
    logger.info(f"Private mode: {PRIVATE}")
    logger.info(f"Detector loaded: {detector_loaded}")
    logger.info(f"Detection threshold: {DETECTION_THRESHOLD}")
    logger.info(f"Classification threshold: {CLASSIFICATION_THRESHOLD}")
    logger.info(f"Max detections: {MAX_DETECTIONS}")
    logger.info("🚀 In-memory processing enabled - no temp files created")
    
    # Force garbage collection and small delay before starting Flask
    gc.collect()
    time.sleep(2)  # Increased delay for GPU initialization
    
    # Run with single-threaded mode to prevent multi-threading GPU conflicts
    app.run(
        host=host,
        port=PORT,
        debug=False,
        use_reloader=False,
        threaded=False  # Single-threaded to prevent GPU conflicts
    )
