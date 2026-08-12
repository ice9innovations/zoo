#!/usr/bin/env python3
"""
Detectron2 Object Detection and Instance Segmentation REST API Service
Coordination service for Detectron2 framework integration.
"""

# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import time
import warnings
import json
import requests
import uuid
import logging
import threading
import random
from typing import List, Dict, Any, Optional

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from urllib.parse import urlparse
from PIL import Image

# Simple emoji lookup - no complex dependencies needed

# Load environment variables first
load_dotenv()

# Configuration for GitHub raw file downloads (optional - fallback to local config)
TIMEOUT = float(os.getenv('TIMEOUT', '10.0'))  # Default 10 seconds for GitHub requests
AUTO_UPDATE = os.getenv('AUTO_UPDATE', 'True').lower() == 'true'  # Enable/disable GitHub downloads

# Detectron2 imports (assumes Detectron2 repo is installed)
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

# Local predictor module (exists in Detectron2 repo)
from detectron2_analyzer import Detectron2Analyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = './uploads'
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
CONFIDENCE_THRESHOLD = float(CONFIDENCE_THRESHOLD_STR) if CONFIDENCE_THRESHOLD_STR else 0.5
IMAGE_SIZE = 160

# Performance optimization flags
USE_HALF_PRECISION = True  # Testing FP16 for VRAM optimization

# Configuration file paths (adjust based on Detectron2 installation)
#CONFIG_FILES = [
#    "./mask_rcnn_R_50_FPN_3x.yaml",  # Local config file
#    "./Base-RCNN-FPN.yaml",          # Alternative local config
#    "/home/sd/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
#    "/opt/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
#    "./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
#]

CONFIG_FILE = "./configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"

load_dotenv()

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
            logger.info(f"🔄 Detectron2: Loading fresh emoji mappings from GitHub: {github_url}")
            response = requests.get(github_url, timeout=TIMEOUT)
            response.raise_for_status()
            data = response.json()
            
            # Cache to disk for future offline use
            try:
                with open(local_cache_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                logger.info(f"💾 Detectron2: Cached emoji mappings to {local_cache_path}")
            except Exception as cache_error:
                logger.warning(f"⚠️  Detectron2: Failed to cache emoji mappings: {cache_error}")
            
            emoji_mappings = data
            logger.info("✅ Detectron2: Successfully loaded emoji mappings from GitHub")
            return
        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️  Detectron2: Failed to load emoji mappings from GitHub: {e}")
            logger.info("🔄 Detectron2: Falling back to local cache due to GitHub failure")
    else:
        logger.info("🔄 Detectron2: AUTO_UPDATE disabled, using local cache only")
        
    # Fallback to local cached file
    try:
        logger.info(f"🔄 Detectron2: Loading emoji mappings from local cache: {local_cache_path}")
        with open(local_cache_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        emoji_mappings = data
        logger.info("✅ Detectron2: Successfully loaded emoji mappings from local cache")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"❌ Detectron2: Failed to load local emoji mappings from {local_cache_path}: {e}")
        if AUTO_UPDATE:
            raise Exception(f"Failed to load emoji mappings from both GitHub and local cache: {e}")
        else:
            raise Exception(f"Failed to load emoji mappings - AUTO_UPDATE disabled and no local cache available. Set AUTO_UPDATE=True or provide emoji_mappings.json in detectron2 directory: {e}")

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

# Load emoji mappings on startup
load_emoji_mappings()

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Global variables
analyzer = None
coco_classes = []

# IoU threshold for filtering overlapping detections  
# Lowered from 0.45 to 0.3 to catch more overlapping detections
IOU_THRESHOLD = 0.3

# ML processing functions moved to analyzer - these are now utility functions

# Image resizing moved to analyzer

def find_config_file() -> Optional[str]:
    """Check for required Detectron2 config file"""
    if os.path.exists(CONFIG_FILE):
        logger.info(f"Found config file: {CONFIG_FILE}")
        return CONFIG_FILE
    else:
        logger.error(f"Required config file not found: {CONFIG_FILE}")
        return None

def load_coco_classes() -> bool:
    """Load COCO class names"""
    global coco_classes
    try:
        with open('coco_classes.txt', 'r') as f:
            coco_classes = f.read().splitlines()
        logger.info(f"Loaded {len(coco_classes)} COCO classes")
        return True
    except Exception as e:
        logger.error(f"Failed to load COCO classes: {e}")
        return False


def initialize_detectron2_analyzer() -> bool:
    """Initialize Detectron2 analyzer once at startup"""
    global analyzer
    try:
        logger.info("Initializing Detectron2 Analyzer...")
        
        # Find config file
        config_file = find_config_file()
        if not config_file:
            logger.error("No Detectron2 config file found")
            return False
        
        analyzer = Detectron2Analyzer(
            config_file=config_file,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            coco_classes=coco_classes,
            use_half_precision=USE_HALF_PRECISION
        )
        
        logger.info("✅ Detectron2 Analyzer initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error initializing Detectron2 Analyzer: {str(e)}")
        return False

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

def is_allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def is_raw_image_request() -> bool:
    return (request.content_type or '').split(';', 1)[0].strip().lower() in RAW_IMAGE_CONTENT_TYPES


def validate_file_size(file_path: str) -> bool:
    """Validate file size"""
    try:
        return os.path.getsize(file_path) <= MAX_FILE_SIZE
    except OSError:
        return False

# Detection processing moved to analyzer

def process_image_for_detection(image: Image.Image) -> Dict[str, Any]:
    """
    Main processing function - takes PIL Image, returns detection data
    This orchestrates the analyzer call - pure business logic orchestration
    """
    try:
        if not analyzer:
            return {
                "success": False,
                "error": "Detectron2 analyzer not loaded"
            }
        
        # Use analyzer for core ML processing
        result = analyzer.analyze_from_pil_image(image)
        
        # Add emoji information to detections
        detections = result.get('detections', [])
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

def create_detectron_response(data: Dict[str, Any], processing_time: float = None) -> Dict[str, Any]:
    """Create standardized detectron2 response"""
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
        
        # Add bbox if present
        if bbox:
            prediction["bbox"] = {
                "x": bbox.get('x', 0),
                "y": bbox.get('y', 0),
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
        "service": "detectron2",
        "status": "success",
        "predictions": predictions,
        "metadata": {
            "processing_time": round(processing_time, 3),
            "model_info": {
                "framework": "Facebook AI Research"
            }
        }
    }

# Legacy detect_objects function removed - functionality moved to analyzer

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
print("Detectron2 service: CORS enabled for direct browser communication")

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
                'service': 'detectron2',
                'error': 'Detectron2 analyzer not initialized'
            }), 503
        
        # Test actual functionality
        if not analyzer.is_healthy():
            return jsonify({
                'status': 'unhealthy', 
                'service': 'detectron2',
                'error': 'Detectron2 analyzer non-functional'
            }), 503
        
        return jsonify({
            'status': 'healthy',
            'service': 'detectron2',
            'capabilities': ['object_detection', 'instance_segmentation', 'bbox_extraction'],
            'models': {
                'object_detection': {
                    'status': 'ready',
                    'framework': 'Detectron2',
                    'confidence_threshold': CONFIDENCE_THRESHOLD,
                    'classes_loaded': len(coco_classes)
                }
            },
            'supported_classes': len(coco_classes),
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
            'service': 'detectron2',
            'error': str(e)
        }), 500

@app.route('/classes', methods=['GET'])
def get_classes():
    """Get supported object classes"""
    return jsonify({
        "classes": coco_classes,
        "total_classes": len(coco_classes),
        "framework": "Detectron2"
    })



@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    """Unified analyze endpoint - orchestrates input handling and processing"""
    start_time = time.time()
    
    def error_response(message: str, status_code: int = 400):
        return jsonify({
            "service": "detectron2",
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
        response = create_detectron_response(processing_result["data"])
        
        return jsonify(response)
        
    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        logger.error(f"Analyze API error: {e}")
        return error_response(f"Internal error: {str(e)}", 500)

@app.route('/v3/analyze', methods=['GET', 'POST'])
def analyze_v3_compat():
    """V3 compatibility - calls new analyze function directly"""
    try:
        return analyze()
    except Exception as e:
        logger.error(f"V3 compatibility error: {e}")
        return jsonify({
            "service": "detectron2", 
            "status": "error",
            "predictions": [],
            "error": {"message": f"V3 endpoint error: {str(e)}"},
            "metadata": {"processing_time": 0}
        }), 500

@app.route('/v2/analyze_file', methods=['GET'])
def analyze_file_v2_compat():
    """V2 file compatibility - translate parameters to new analyze format"""
    file_path = request.args.get('file_path')
    
    if file_path:
        new_args = {'file': file_path}
        with app.test_request_context('/analyze', query_string=new_args):
            return analyze()
    else:
        with app.test_request_context('/analyze'):
            return analyze()

@app.route('/v2/analyze', methods=['GET'])
def analyze_v2_compat():
    """V2 compatibility - translate parameters to new analyze format"""
    image_url = request.args.get('image_url')
    
    if image_url:
        # Parameter translation
        new_args = request.args.copy().to_dict()
        new_args['url'] = image_url
        if 'image_url' in new_args:
            del new_args['image_url']
        
        # Call new analyze with translated parameters
        with app.test_request_context('/analyze', query_string=new_args):
            return analyze()
    else:
        # Let new analyze handle validation errors
        with app.test_request_context('/analyze'):
            return analyze()

@app.route('/v3/analyze_old', methods=['GET'])
def analyze_v3_old():
    """Legacy V3 alias for the unified in-memory analyze endpoint."""
    return analyze()


if __name__ == '__main__':
    # Set multiprocessing start method for Detectron2
    mp.set_start_method("spawn", force=True)
    
    # Initialize services
    logger.info("Starting Detectron2 service...")
    
    coco_loaded = load_coco_classes()
    model_loaded = initialize_detectron2_analyzer()
    
    # Using local emoji file - no external dependencies
    
    if not model_loaded:
        logger.error("Failed to load Detectron2 analyzer.")
        logger.error("Please ensure Detectron2 is installed and config files are available.")
        logger.error("Service cannot function without analyzer. Exiting.")
        exit(1)
        
    if not coco_loaded:
        logger.error("Failed to load COCO classes.")
        logger.error("Service cannot function without class definitions. Exiting.")
        exit(1)
        
    
    # Always use 0.0.0.0 to allow external connections
    host = "0.0.0.0"
    
    logger.info(f"Starting Detectron2 service on {host}:{PORT}")
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
