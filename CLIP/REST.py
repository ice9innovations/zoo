from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from PIL import Image
import sys
import os
import os.path
import requests
import json
import re
import uuid
import logging
import random
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse

from dotenv import load_dotenv
load_dotenv()

from clip_analyzer import ClipAnalyzer

# Configuration for GitHub raw file downloads (optional - fallback to local config)
API_TIMEOUT = float(os.getenv('API_TIMEOUT', '10.0'))  # Default 10 seconds for GitHub requests
AUTO_UPDATE = os.getenv('AUTO_UPDATE', 'True').lower() == 'true'  # Enable/disable GitHub downloads

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

# Validate critical configuration
if not PRIVATE_STR:
    raise ValueError("PRIVATE environment variable is required")
if not PORT_STR:
    raise ValueError("PORT environment variable is required")

# Convert to appropriate types
PRIVATE = PRIVATE_STR.lower() == 'true'
PORT = int(PORT_STR)

# Prediction filtering configuration - easily adjustable
DEFAULT_CONFIDENCE_THRESHOLD = 0.00  #  only return reasonably confident predictions
DEFAULT_MAX_PREDICTIONS = 100         # Safety cap on number of predictions

CONFIDENCE_THRESHOLD_STR = os.getenv('CLIP_CONFIDENCE_THRESHOLD')
MAX_PREDICTIONS_STR = os.getenv('CLIP_MAX_PREDICTIONS')

# Validate CLIP configuration
if not CONFIDENCE_THRESHOLD_STR:
    raise ValueError("CLIP_CONFIDENCE_THRESHOLD environment variable is required")
if not MAX_PREDICTIONS_STR:
    raise ValueError("CLIP_MAX_PREDICTIONS environment variable is required")

# Convert to appropriate types
CONFIDENCE_THRESHOLD = float(CONFIDENCE_THRESHOLD_STR)
MAX_PREDICTIONS = int(MAX_PREDICTIONS_STR)

# Label file configuration - automatically loads all .txt files from labels folder
LABELS_FOLDER = './labels'

# CLIP model configuration - Options: ViT-B/32, ViT-L/14, ViT-L/14@336px
CLIP_MODEL = 'ViT-L/14'  # Larger model for better discrimination (~6-8GB VRAM)
#CLIP_MODEL = 'ViT-B/32'  # Smaller model (~4-6GB VRAM)

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

TEXT_EMBEDDINGS_CACHE_FILE = './text_embeddings_cache.pkl'


# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    device = torch.device("mps")

logger.info(f"Using device: {device}")

# Load emoji mappings with GitHub-first approach
emoji_mappings = {}

def load_emoji_mappings():
    """Load emoji mappings from GitHub raw files with local caching"""
    global emoji_mappings
    local_cache_path = os.path.join(os.path.dirname(__file__), 'emoji_mappings.json')
    
    # Try GitHub raw file first if AUTO_UPDATE is enabled
    if AUTO_UPDATE:
        github_url = "https://raw.githubusercontent.com/ice9innovations/animal-farm/refs/heads/main/config/emoji_mappings.json"
        
        try:
            logger.info(f"🔄 CLIP: Loading fresh emoji mappings from GitHub: {github_url}")
            response = requests.get(github_url, timeout=API_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            
            # Cache to disk for future offline use
            try:
                with open(local_cache_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                logger.info(f"💾 CLIP: Cached emoji mappings to {local_cache_path}")
            except Exception as cache_error:
                logger.warning(f"⚠️  CLIP: Failed to cache emoji mappings: {cache_error}")
            
            emoji_mappings = data
            logger.info("✅ CLIP: Successfully loaded emoji mappings from GitHub")
            return
        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️  CLIP: Failed to load emoji mappings from GitHub: {e}")
            logger.info("🔄 CLIP: Falling back to local cache due to GitHub failure")
    else:
        logger.info("🔄 CLIP: AUTO_UPDATE disabled, using local cache only")
        
    # Fallback to local cached file
    try:
        logger.info(f"🔄 CLIP: Loading emoji mappings from local cache: {local_cache_path}")
        with open(local_cache_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        emoji_mappings = data
        logger.info("✅ CLIP: Successfully loaded emoji mappings from local cache")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"❌ CLIP: Failed to load local emoji mappings from {local_cache_path}: {e}")
        if AUTO_UPDATE:
            raise Exception(f"Failed to load emoji mappings from both GitHub and local cache: {e}")
        else:
            raise Exception(f"Failed to load emoji mappings - AUTO_UPDATE disabled and no local cache available. Set AUTO_UPDATE=True or provide emoji_mappings.json in CLIP directory: {e}")

def get_emoji(concept: str) -> Optional[str]:
    """Get emoji for a single concept"""
    if not concept:
        return None
    # Strip punctuation before processing
    import string
    concept_clean = concept.translate(str.maketrans('', '', string.punctuation))
    concept_clean = concept_clean.lower().strip().replace(' ', '_')
    return emoji_mappings.get(concept_clean)


def check_shiny():
    """Check if this detection should be shiny (1/2500 chance)"""
    roll = random.randint(1, 2500)
    is_shiny = roll == 1
    return is_shiny, roll

# Load emoji mappings on startup
load_emoji_mappings()



# Global analyzer instance
analyzer = None

def initialize_analyzer() -> bool:
    """Initialize analyzer once at startup - fail fast"""
    global analyzer
    analyzer = ClipAnalyzer(
        model_name=CLIP_MODEL,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        max_predictions=MAX_PREDICTIONS,
        labels_folder=LABELS_FOLDER,
        cache_file=TEXT_EMBEDDINGS_CACHE_FILE
    )
    return analyzer.initialize()


def lookup_emoji(tag: str, score: float) -> List[Dict[str, Any]]:
    """Look up emoji for a given tag with score using local emoji service (optimized - no HTTP requests)"""
    # Convert tag to string - let get_emoji() handle all normalization
    original_tag = str(tag).strip()
    
    try:
        # Use local emoji service instead of HTTP requests
        emoji = get_emoji(original_tag)
        if emoji:
            logger.info(f"Local emoji service: '{original_tag}' → {emoji} (confidence: {score:.3f})")
            return [{
                "keyword": original_tag,
                "emoji": emoji,
                "confidence": round(float(score), 3)
            }]
        
        logger.debug(f"Local emoji service: no emoji found for '{original_tag}'")
        return []
        
    except Exception as e:
        logger.warning(f"Local emoji service lookup failed for '{original_tag}': {e}")
        return []



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

def process_image_for_classification(image: Image.Image) -> Dict[str, Any]:
    """
    Main processing function - takes PIL Image, returns classification data
    This is the core business logic, separated from HTTP concerns
    """
    try:
        # Use analyzer instead of direct ML logic
        result = analyzer.analyze_similarity_from_array(image)
        
        if not result.get('success'):
            return {
                "success": False,
                "error": result.get('error', 'Classification failed')
            }
        
        # Extract predictions and run emoji lookup
        predictions = result.get('predictions', [])
        all_emoji_matches = []
        
        # Look up emoji for each prediction (same as original logic)
        for prediction in predictions:
            label = prediction.get('label', '')
            confidence = prediction.get('confidence', 0)
            
            try:
                emoji_match = lookup_emoji(label, confidence)
                if emoji_match:
                    all_emoji_matches.extend(emoji_match)
                else:
                    logger.debug(f"No emoji found for label '{label}'")
            except RuntimeError as e:
                logger.error(f"Mirror Stage service failure for '{label}': {e}")
                raise RuntimeError(f"Classification failed due to Mirror Stage service failure: {e}")
        
        return {
            "success": True,
            "predictions": predictions,
            "emoji_matches": all_emoji_matches
        }
        
    except Exception as e:
        logger.error(f"Error processing image for classification: {e}")
        return {
            "success": False,
            "error": f"Processing failed: {str(e)}"
        }

def create_clip_response(predictions: List[Dict], emoji_matches: List[Dict], processing_time: float) -> Dict[str, Any]:
    """Create standardized CLIP response with emoji mappings"""
    # Create emoji map for quick lookup
    emoji_map = {}
    for match in emoji_matches:
        emoji_map[match["keyword"]] = match["emoji"]
    
    # Create unified prediction format
    formatted_predictions = []
    for item in predictions:
        is_shiny, shiny_roll = check_shiny()
        
        prediction = {
            "label": item.get('label', ''),
            "confidence": round(float(item.get('confidence', 0)), 3)
        }
        
        # Add shiny flag only for shiny detections
        if is_shiny:
            prediction["shiny"] = True
            logger.info(f"✨ SHINY {item.get('label', '').upper()} DETECTED! Roll: {shiny_roll} ✨")
        
        # Add emoji if found
        label = item.get('label', '')
        if label in emoji_map:
            prediction["emoji"] = emoji_map[label]
        
        formatted_predictions.append(prediction)
    
    return {
        "service": "clip",
        "status": "success",
        "predictions": formatted_predictions,
        "metadata": {
            "processing_time": round(processing_time, 3),
            "model_info": {
                "framework": "OpenAI"
            }
        }
    }

def handle_image_input(url: str = None, file: str = None) -> Dict[str, Any]:
    """
    Handle image input from either URL or file path - pure in-memory processing
    Returns: {"success": bool, "image": PIL.Image, "error": str}
    """
    from io import BytesIO
    
    # Validate input - exactly one parameter must be provided
    if not url and not file:
        return {"success": False, "error": "Must provide either url or file parameter"}
    
    if url and file:
        return {"success": False, "error": "Cannot provide both url and file parameters"}
    
    # Handle URL input - download directly to memory
    if url:
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
            
            image = Image.open(BytesIO(response.content)).convert('RGB')
            return {"success": True, "image": image, "error": None}
            
        except Exception as e:
            logger.error(f"Error processing image URL {url}: {e}")
            return {"success": False, "error": f"Failed to process image URL: {str(e)}"}
    
    # Handle file path input - read directly to memory
    elif file:
        try:
            if not os.path.exists(file):
                return {"success": False, "error": f"File not found: {file}"}
            
            if not is_allowed_file(file):
                return {"success": False, "error": "File type not allowed"}
            
            if not validate_file_size(file):
                return {"success": False, "error": "File too large"}
            
            image = Image.open(file).convert('RGB')
            return {"success": True, "image": image, "error": None}
            
        except Exception as e:
            logger.error(f"Error processing image file {file}: {e}")
            return {"success": False, "error": f"Failed to process image file: {str(e)}"}



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
print("CLIP service: CORS enabled for direct browser communication")

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
    model_status = "loaded" if analyzer and analyzer.model else "not_loaded"
    return jsonify({
        "status": "healthy",
        "model_status": model_status,
        "device": str(device),
        "num_labels": len(analyzer.labels) if analyzer and analyzer.labels else 0
    })

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    """Unified analyze endpoint - orchestrates input handling and processing"""
    import time
    from io import BytesIO
    start_time = time.time()
    
    def error_response(message: str, status_code: int = 400):
        return jsonify({
            "service": "clip",
            "status": "error",
            "predictions": [],
            "error": {"message": message},
            "metadata": {"processing_time": round(time.time() - start_time, 3)}
        }), status_code
    
    try:
        image = None
        
        # Step 1: Get image into memory from any source
        if request.method == 'POST' and is_raw_image_request():
            from io import BytesIO
            file_data = request.get_data(cache=False)
            if not file_data:
                return error_response("No image body provided")
            if len(file_data) > MAX_FILE_SIZE:
                return error_response("File too large")
            try:
                image = Image.open(BytesIO(file_data)).convert('RGB')
            except Exception as e:
                return error_response(f"Failed to process raw image body: {str(e)}", 500)

        elif request.method == 'POST' and 'file' in request.files:
            # Handle file upload - direct to memory
            uploaded_file = request.files['file']
            if uploaded_file.filename == '':
                return error_response("No file selected")
            
            if not is_allowed_file(uploaded_file.filename):
                return error_response("File type not allowed")
            
            # Read directly into memory
            file_data = uploaded_file.read()
            if len(file_data) > MAX_FILE_SIZE:
                return error_response("File too large")
            
            image = Image.open(BytesIO(file_data)).convert('RGB')
        
        else:
            # Handle URL or file parameter
            url = request.args.get('url')
            file = request.args.get('file')
            
            if not url and not file:
                return error_response("Must provide either url or file parameter, or POST a file")
            
            if url and file:
                return error_response("Cannot provide both url and file parameters")
            
            # Handle URL or file parameter using shared helper
            image_result = handle_image_input(url=url, file=file)
            
            if not image_result["success"]:
                return error_response(image_result["error"])
            
            image = image_result["image"]
        
        # Step 2: Process the image (unified processing path)
        processing_result = process_image_for_classification(image)
        
        # Step 3: Handle processing result
        if not processing_result["success"]:
            return error_response(processing_result["error"], 500)
        
        # Step 4: Create response
        response = create_clip_response(
            processing_result["predictions"],
            processing_result["emoji_matches"],
            time.time() - start_time
        )
        
        return jsonify(response)
        
    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        logger.error(f"Analyze API error: {e}")
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

@app.route('/score', methods=['GET', 'POST'])
def score_caption():
    """Score caption similarity against image using CLIP"""
    import time
    from io import BytesIO
    start_time = time.time()
    
    try:
        # Get input parameters - support both GET and POST
        caption = None
        image = None
        
        if request.method == 'POST':
            if is_raw_image_request():
                file_data = request.get_data(cache=False)
                caption = request.args.get('caption') or request.headers.get('X-Ice9-Caption')
                if not file_data:
                    return jsonify({
                        "service": "clip",
                        "status": "error",
                        "similarity_score": None,
                        "error": {"message": "No image body provided"},
                        "metadata": {"processing_time": round(time.time() - start_time, 3)}
                    }), 400
                if len(file_data) > MAX_FILE_SIZE:
                    return jsonify({
                        "service": "clip",
                        "status": "error",
                        "similarity_score": None,
                        "error": {"message": "File too large"},
                        "metadata": {"processing_time": round(time.time() - start_time, 3)}
                    }), 400
                image = Image.open(BytesIO(file_data)).convert('RGB')

            # Handle multipart file upload (from caption scoring worker)
            elif 'file' in request.files:
                uploaded_file = request.files['file']
                caption = request.form.get('caption')
                
                if uploaded_file.filename == '':
                    return jsonify({
                        "service": "clip",
                        "status": "error",
                        "similarity_score": None,
                        "error": {"message": "No file selected"},
                        "metadata": {"processing_time": round(time.time() - start_time, 3)}
                    }), 400
                
                if not is_allowed_file(uploaded_file.filename):
                    return jsonify({
                        "service": "clip",
                        "status": "error",
                        "similarity_score": None,
                        "error": {"message": "File type not allowed"},
                        "metadata": {"processing_time": round(time.time() - start_time, 3)}
                    }), 400
                
                # Read file directly into memory
                file_data = uploaded_file.read()
                if len(file_data) > MAX_FILE_SIZE:
                    return jsonify({
                        "service": "clip",
                        "status": "error",
                        "similarity_score": None,
                        "error": {"message": "File too large"},
                        "metadata": {"processing_time": round(time.time() - start_time, 3)}
                    }), 400
                
                image = Image.open(BytesIO(file_data)).convert('RGB')
            
            # Handle JSON or form data
            elif request.is_json:
                data = request.get_json()
                caption = data.get('caption')
                url = data.get('url')
                file = data.get('file')
            else:
                caption = request.form.get('caption') or request.args.get('caption')
                url = request.form.get('url') or request.args.get('url')  
                file = request.form.get('file') or request.args.get('file')
        else:
            # For GET, use query parameters
            caption = request.args.get('caption')
            url = request.args.get('url')
            file = request.args.get('file')
        
        # Validate caption parameter
        if not caption or not isinstance(caption, str) or not caption.strip():
            return jsonify({
                "service": "clip",
                "status": "error",
                "similarity_score": None,
                "error": {"message": "Must provide non-empty caption string"},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 400
        
        caption = caption.strip()
        
        # If no image loaded from multipart upload, fail
        if image is None:
            return jsonify({
                "service": "clip",
                "status": "error",
                "similarity_score": None,
                "error": {"message": "Must provide image file via multipart upload"},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 400
        
        # Check analyzer availability
        if analyzer is None or analyzer.model is None:
            return jsonify({
                "service": "clip",
                "status": "error",
                "similarity_score": None,
                "error": {"message": "CLIP analyzer not initialized"},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 500
        
        # Compute similarity score using analyzer
        similarity_score = analyzer.compute_caption_similarity(image, caption)
        
        if similarity_score is None:
            return jsonify({
                "service": "clip",
                "status": "error",
                "similarity_score": None,
                "error": {"message": "Failed to compute similarity score"},
                "metadata": {"processing_time": round(time.time() - start_time, 3)}
            }), 500
        
        return jsonify({
            "service": "clip",
            "status": "success",
            "similarity_score": round(float(similarity_score), 3),
            "caption": caption,
            "image_source": "upload",
            "metadata": {
                "processing_time": round(time.time() - start_time, 3),
                "model_info": {
                    "framework": "OpenAI",
                    "model": CLIP_MODEL
                }
            }
        })
        
    except Exception as e:
        logger.error(f"V3 caption scoring error: {e}")
        return jsonify({
            "service": "clip",
            "status": "error",
            "similarity_score": None,
            "error": {"message": f"Internal error: {str(e)}"},
            "metadata": {"processing_time": round(time.time() - start_time, 3)}
        }), 500

@app.route('/v3/score', methods=['GET', 'POST'])
def score_caption_v3_compat():
    """V3 compatibility - redirect to new score endpoint"""
    if request.method == 'POST':
        # Forward POST request with data
        return score_caption()
    else:
        # Forward GET request with query string
        with app.test_request_context('/score', query_string=request.args):
            return score_caption()

# V2 Compatibility Routes - Translate parameters and call V3
@app.route('/v2/analyze_file/', methods=['GET'])
@app.route('/v2/analyze_file', methods=['GET'])
def analyze_file_v2_compat():
    """V2 file compatibility - translate parameters to V3 format"""
    # Get V2 parameter
    file_path = request.args.get('file_path')
    
    if file_path:
        # Create new request args with V3 parameter name
        new_args = {'file': file_path}
        
        # Create a mock request object for new analyze endpoint
        with app.test_request_context('/analyze', query_string=new_args):
            return analyze()
    else:
        # No parameters - let analyze handle the error
        with app.test_request_context('/analyze'):
            return analyze()

@app.route('/v2/analyze/', methods=['GET'])
@app.route('/v2/analyze', methods=['GET'])
def analyze_v2_compat():
    """V2 compatibility - translate parameters to V3 format"""
    # Get V2 parameter
    image_url = request.args.get('image_url')
    
    if image_url:
        # Create new request args with V3 parameter name
        new_args = request.args.copy()
        new_args = new_args.to_dict()
        new_args['url'] = image_url
        del new_args['image_url']
        
        # Create a mock request object for new analyze endpoint
        with app.test_request_context('/analyze', query_string=new_args):
            return analyze()
    else:
        # No parameters - let analyze handle the error
        with app.test_request_context('/analyze'):
            return analyze()


if __name__ == '__main__':
    # Initialize analyzer and emoji data
    logger.info("Starting CLIP service...")
    
    analyzer_loaded = initialize_analyzer()
    
    if not analyzer_loaded:
        logger.error("Failed to load CLIP analyzer. Service will run but classification will fail.")
        logger.error("Please ensure CLIP is installed: pip install clip-by-openai")
        
    
    # Determine host based on private mode
    host = "127.0.0.1" if PRIVATE else "0.0.0.0"
    
    logger.info(f"Starting CLIP service on {host}:{PORT}")
    logger.info(f"Private mode: {PRIVATE}")
    logger.info(f"CLIP model: {CLIP_MODEL}")
    logger.info(f"Analyzer loaded: {analyzer_loaded}")
    if analyzer and analyzer.labels:
        logger.info(f"Classification labels: {len(analyzer.labels)}")
    
    app.run(
        host=host,
        port=PORT,
        debug=False,
        use_reloader=False,
        threaded=True
    )
