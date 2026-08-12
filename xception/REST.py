#!/usr/bin/env python3
"""
Xception Image Classification REST API Service
Coordination service for Google's Xception model for ImageNet classification.
"""

import os
import json
import uuid
import logging
import random
import time
from typing import List, Dict, Any, Optional

import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from urllib.parse import urlparse
from PIL import Image

# Import our analyzer
from xception_analyzer import XceptionAnalyzer

# Simple emoji lookup - no complex dependencies needed

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Load environment variables as strings first
AUTO_UPDATE_STR = os.getenv('AUTO_UPDATE', 'true')
PORT_STR = os.getenv('PORT')
PRIVATE_STR = os.getenv('PRIVATE')
CONFIDENCE_THRESHOLD_STR = os.getenv('CONFIDENCE_THRESHOLD')

# Validate required environment variables
if not PORT_STR:
    raise ValueError("PORT environment variable is required")
if not PRIVATE_STR:
    raise ValueError("PRIVATE environment variable is required")

# Convert to appropriate types after validation
AUTO_UPDATE = AUTO_UPDATE_STR.lower() == 'true'
PORT = int(PORT_STR)
PRIVATE = PRIVATE_STR.lower() == 'true'
CONFIDENCE_THRESHOLD = float(CONFIDENCE_THRESHOLD_STR) if CONFIDENCE_THRESHOLD_STR else 0.15


# GPU configuration is now handled by XceptionAnalyzer

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
IMAGE_SIZE = 160


# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables
analyzer = None


def is_raw_image_request() -> bool:
    return (request.content_type or '').split(';', 1)[0].strip().lower() in RAW_IMAGE_CONTENT_TYPES

# Load emoji mappings from local JSON file
emoji_mappings = {}

def load_emoji_mappings():
    """Load emoji mappings from GitHub, fall back to local cache"""
    global emoji_mappings
    
    github_url = "https://raw.githubusercontent.com/ice9innovations/animal-farm/refs/heads/main/config/emoji_mappings.json"
    local_cache_path = 'emoji_mappings.json'

    if AUTO_UPDATE:
        try:
            logger.info(f"🔄 Xception: Loading emoji mappings from GitHub: {github_url}")
            response = requests.get(github_url, timeout=10.0)
            response.raise_for_status()
            
            # Save to local cache (preserve emoji characters)
            with open(local_cache_path, 'w', encoding='utf-8') as f:
                json.dump(response.json(), f, indent=2, ensure_ascii=False)
            
            emoji_mappings = response.json()
            logger.info(f"✅ Xception: Loaded emoji mappings from GitHub and cached locally ({len(emoji_mappings)} entries)")
            return
        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️  Xception: Failed to load emoji mappings from GitHub ({e}), falling back to local cache")
    
    # Fall back to local cache
    try:
        with open(local_cache_path, 'r') as f:
            emoji_mappings = json.load(f)
            logger.info(f"✅ Xception: Successfully loaded emoji mappings from local cache ({len(emoji_mappings)} entries)")
    except Exception as local_error:
        logger.error(f"❌ Xception: Failed to load local emoji mappings: {local_error}")
        logger.warning("⚠️  Xception: No emoji mappings available - emojis will be None")
        emoji_mappings = {}

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

# ImageNet classes are handled by TensorFlow's decode_predictions internally
# No separate loading required


def lookup_emoji(class_name: str) -> Optional[str]:
    """Look up emoji for a given class name using local emoji service (optimized - no HTTP requests)"""
    clean_name = class_name.lower().strip().replace(" ", "_")
    
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

def initialize_xception_analyzer() -> bool:
    """Initialize Xception analyzer"""
    global analyzer
    try:
        logger.info(f"Initializing Xception analyzer with confidence threshold: {CONFIDENCE_THRESHOLD}")
        analyzer = XceptionAnalyzer(
            confidence_threshold=CONFIDENCE_THRESHOLD,
            input_size=299
        )
        return analyzer.initialize()
        
    except Exception as e:
        logger.error(f"Failed to initialize Xception analyzer: {e}")
        return False

def process_image_for_classification(image: Image.Image) -> dict:
    """
    Main processing function - takes PIL Image, returns classification data
    This is the core business logic, separated from HTTP concerns
    Uses pure in-memory processing
    """
    if not analyzer:
        return {
            "success": False,
            "error": "Xception analyzer not initialized"
        }
    
    try:
        # Use analyzer to classify image
        result = analyzer.analyze_image(image)
        
        if not result["success"]:
            return result
        
        # Add emoji lookup to classifications
        classifications = result["data"]["classifications"]
        seen_emojis = set()
        filtered_classifications = []
        
        for classification in classifications:
            try:
                emoji = lookup_emoji(classification["class_name"])
                
                # Only include if we haven't seen this emoji before
                if emoji not in seen_emojis:
                    classification["emoji"] = emoji
                    filtered_classifications.append(classification)
                    seen_emojis.add(emoji)
            except RuntimeError as e:
                return {
                    "success": False,
                    "error": f"Classification failed due to emoji lookup failure: {e}"
                }
        
        # Update the result with filtered classifications
        result["data"]["classifications"] = filtered_classifications
        result["data"]["total_classifications"] = len(filtered_classifications)
        
        return result
        
    except Exception as e:
        logger.error(f"Classification processing error: {str(e)}")
        return {
            "success": False,
            "error": f"Classification failed: {str(e)}"
        }

def create_xception_response(data: dict, processing_time: float) -> dict:
    """Create standardized xception response"""
    classifications = data.get('classifications', [])
    
    # Create unified prediction format
    predictions = []
    for classification in classifications:
        is_shiny, shiny_roll = check_shiny()
        
        prediction = {
            "confidence": round(float(classification.get('confidence', 0)), 3),
            "label": classification.get('class_name', '')
        }
        
        # Add shiny flag only for shiny detections
        if is_shiny:
            prediction["shiny"] = True
            logger.info(f"✨ SHINY {classification.get('class_name', '').upper()} DETECTED! Roll: {shiny_roll} ✨")
        
        # Add emoji if present
        if classification.get('emoji'):
            prediction["emoji"] = classification['emoji']
        
        predictions.append(prediction)
    
    return {
        "service": "xception",
        "status": "success",
        "predictions": predictions,
        "metadata": {
            "processing_time": round(processing_time, 3),
            "model_info": {
                "framework": "TensorFlow"
            }
        }
    }

def is_allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_file_size(file_path: str) -> bool:
    """Validate file size"""
    try:
        return os.path.getsize(file_path) <= MAX_FILE_SIZE
    except OSError:
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
print("Xception service: CORS enabled for direct browser communication")

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
    if not analyzer:
        return jsonify({
            "status": "unhealthy",
            "reason": "Xception analyzer not initialized",
            "framework": "TensorFlow"
        }), 503
    
    model_info = analyzer.get_model_info()
    if not model_info["model_loaded"]:
        return jsonify({
            "status": "unhealthy",
            "reason": "Xception model not loaded",
            "framework": "TensorFlow"
        }), 503
    
    return jsonify({
        "status": "healthy",
        "model_status": "loaded",
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "imagenet_classes_loaded": "handled_by_tensorflow",
        "framework": "TensorFlow",
        "model": "Xception"
    })

@app.route('/classes', methods=['GET'])
def get_classes():
    """Get supported ImageNet classes"""
    return jsonify({
        "classes": "handled_by_tensorflow",  # Classes handled by decode_predictions
        "total_classes": 1000,
        "framework": "TensorFlow",
        "model": "Xception"
    })

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    """Unified analyze endpoint - orchestrates input handling and processing"""
    start_time = time.time()
    
    def error_response(message: str, status_code: int = 400):
        return jsonify({
            "service": "xception",
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
        processing_result = process_image_for_classification(image)
        
        # Step 3: Handle processing result
        if not processing_result["success"]:
            return error_response(processing_result["error"], 500)
        
        # Step 4: Create response
        response = create_xception_response(
            processing_result["data"],
            processing_result["processing_time"]
        )
        
        return jsonify(response)
        
    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        logger.error(f"Analyze API error: {e}")
        return error_response(f"Internal error: {str(e)}", 500)

@app.route('/v3/analyze', methods=['GET', 'POST'])
def analyze_v3_compat():
    """V3 compatibility - calls new analyze function directly"""
    return analyze()

@app.route('/v2/analyze_file', methods=['GET'])
def analyze_file_v2():
    """V2 API compatibility endpoint - translates file_path to file parameter"""
    file_path = request.args.get('file_path')
    
    if file_path:
        new_args = {'file': file_path}
        with app.test_request_context('/analyze', query_string=new_args):
            return analyze()
    else:
        with app.test_request_context('/analyze'):
            return analyze()

@app.route('/v2/analyze', methods=['GET'])
def analyze_v2():
    """V2 API compatibility endpoint - translates image_url to url parameter"""
    image_url = request.args.get('image_url')
    
    if image_url:
        new_args = request.args.copy().to_dict()
        new_args['url'] = image_url
        if 'image_url' in new_args:
            del new_args['image_url']
        
        with app.test_request_context('/analyze', query_string=new_args):
            return analyze()
    else:
        with app.test_request_context('/analyze'):
            return analyze()



if __name__ == '__main__':
    # Initialize services
    logger.info("Starting Xception service...")
    
    model_loaded = initialize_xception_analyzer()
    
    if not model_loaded:
        logger.error("Failed to load Xception model.")
        logger.error("Please ensure TensorFlow is installed with ImageNet weights.")
        logger.error("Service cannot function without model. Exiting.")
        exit(1)
    
    # Determine host based on private mode
    host = "127.0.0.1" if PRIVATE else "0.0.0.0"
    
    logger.info(f"Starting Xception service on {host}:{PORT}")
    logger.info(f"Private mode: {PRIVATE}")
    logger.info(f"Model loaded: {model_loaded}")
    logger.info(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    
    app.run(
        host=host,
        port=PORT,
        debug=False,
        use_reloader=False,
        threaded=True
    )
