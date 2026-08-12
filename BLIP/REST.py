import json
import requests
import os
import sys
import uuid
import logging
import random
import nltk
from nltk.tokenize import MWETokenizer
from typing import List, Optional, Dict, Any

# Add current directory to Python path for model imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

# Configuration for GitHub raw file downloads (optional - fallback to local config)
TIMEOUT = float(os.getenv('TIMEOUT', '10.0'))  # Default 10 seconds for GitHub requests
AUTO_UPDATE = os.getenv('AUTO_UPDATE', 'True').lower() == 'true'  # Enable/disable GitHub downloads

import torch
from urllib.parse import urlparse, parse_qs
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from flask import Flask, request, jsonify
from flask_cors import CORS
from blip_analyzer import BlipAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
IMAGE_SIZE = 384
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

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load emoji mappings from local JSON file
emoji_mappings = {}
emoji_tokenizer = None

# Global BLIP analyzer - initialize once at startup
blip_analyzer = None

# Priority overrides for critical ambiguities
PRIORITY_OVERRIDES = {
    'glass': '🥛',        # drinking glass, not eyewear
    'glasses': '👓',      # eyewear (explicit to avoid fallback)
    'wood': '🌲',         # base material form
    'wooden': '🌲',       # morphological variant
    'metal': '🔧',        # base material form  
    'metallic': '🔧',     # morphological variant
}

def load_emoji_mappings():
    """Load emoji mappings from GitHub raw files with local caching"""
    local_cache_path = os.path.join(os.path.dirname(__file__), 'emoji_mappings.json')
    
    # Try GitHub raw file first if AUTO_UPDATE is enabled
    if AUTO_UPDATE:
        github_url = "https://raw.githubusercontent.com/ice9innovations/animal-farm/refs/heads/main/config/emoji_mappings.json"
        
        try:
            logger.info(f"🔄 BLIP: Loading fresh emoji mappings from GitHub: {github_url}")
            response = requests.get(github_url, timeout=TIMEOUT)
            response.raise_for_status()
            data = response.json()
            
            # Cache to disk for future offline use
            try:
                with open(local_cache_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                logger.info(f"💾 BLIP: Cached emoji mappings to {local_cache_path}")
            except Exception as cache_error:
                logger.warning(f"⚠️  BLIP: Failed to cache emoji mappings: {cache_error}")
            
            logger.info("✅ BLIP: Successfully loaded emoji mappings from GitHub")
            return data
        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️  BLIP: Failed to load emoji mappings from GitHub: {e}")
            logger.info("🔄 BLIP: Falling back to local cache due to GitHub failure")
    else:
        logger.info("🔄 BLIP: AUTO_UPDATE disabled, using local cache only")
        
    # Fallback to local cached file
    try:
        logger.info(f"🔄 BLIP: Loading emoji mappings from local cache: {local_cache_path}")
        with open(local_cache_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info("✅ BLIP: Successfully loaded emoji mappings from local cache")
        return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"❌ BLIP: Failed to load local emoji mappings from {local_cache_path}: {e}")
        if AUTO_UPDATE:
            raise Exception(f"Failed to load emoji mappings from both GitHub and local cache: {e}")
        else:
            raise Exception(f"Failed to load emoji mappings - AUTO_UPDATE disabled and no local cache available. Set AUTO_UPDATE=True or provide emoji_mappings.json in BLIP directory: {e}")

def load_mwe_mappings():
    """Load MWE mappings from GitHub raw files with local caching and convert to tuples"""
    local_cache_path = os.path.join(os.path.dirname(__file__), 'mwe.txt')
    mwe_text = []
    
    # Try GitHub raw file first if AUTO_UPDATE is enabled
    if AUTO_UPDATE:
        github_url = "https://raw.githubusercontent.com/ice9innovations/animal-farm/refs/heads/main/config/mwe.txt"
        
        try:
            logger.info(f"🔄 BLIP: Loading fresh multi-word expressions (MWE) mappings from GitHub: {github_url}")
            response = requests.get(github_url, timeout=TIMEOUT)
            response.raise_for_status()
            mwe_text = response.text.splitlines()
            
            # Cache to disk for future offline use
            try:
                with open(local_cache_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                logger.info(f"💾 BLIP: Cached MWE mappings to {local_cache_path}")
            except Exception as cache_error:
                logger.warning(f"⚠️  BLIP: Failed to cache MWE mappings: {cache_error}")
            
            logger.info("✅ BLIP: Successfully loaded MWE mappings from GitHub")
        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️  BLIP: Failed to load MWE mappings from GitHub: {e}")
            logger.info("🔄 BLIP: Falling back to local cache due to GitHub failure")
    else:
        logger.info("🔄 BLIP: AUTO_UPDATE disabled, using local cache only")
        
    # Fallback to local cached file if GitHub failed or AUTO_UPDATE is disabled
    if not mwe_text:
        try:
            logger.info(f"🔄 BLIP: Loading MWE mappings from local cache: {local_cache_path}")
            with open(local_cache_path, 'r', encoding='utf-8') as f:
                mwe_text = f.read().splitlines()
            logger.info("✅ BLIP: Successfully loaded MWE mappings from local cache")
        except FileNotFoundError as e:
            logger.error(f"❌ BLIP: Failed to load local MWE mappings from {local_cache_path}: {e}")
            if AUTO_UPDATE:
                raise Exception(f"Failed to load MWE mappings from both GitHub and local cache: {e}")
            else:
                raise Exception(f"Failed to load MWE mappings - AUTO_UPDATE disabled and no local cache available. Set AUTO_UPDATE=True or provide mwe.txt in BLIP directory: {e}")

    # Convert to tuples for MWETokenizer
    mwe_tuples = []
    for line in mwe_text:
        if line.strip():  # Skip empty lines
            # Convert underscore format to word tuples (e.g., "street_sign" -> ("street", "sign"))
            mwe_tuples.append(tuple(line.strip().replace('_', ' ').split()))
    
    return mwe_tuples

# Load emoji mappings on startup
emoji_mappings = load_emoji_mappings()
mwe_mappings = load_mwe_mappings()

# Initialize MWE tokenizer with the loaded mappings (already converted to tuples)
emoji_tokenizer = MWETokenizer(mwe_mappings, separator='_')

def get_emoji_for_word(word: str) -> str:
    """Get emoji for a single word with morphological variations"""
    if not word:
        return None

    word_clean = word.lower().strip()

    if word_clean in PRIORITY_OVERRIDES:
        return PRIORITY_OVERRIDES[word_clean]

    # Try exact match in main mappings
    if word_clean in emoji_mappings:
        return emoji_mappings[word_clean]

    # Try singular form for common plurals
    if word_clean.endswith('s') and len(word_clean) > 3:
        singular = word_clean[:-1]
        if singular in emoji_mappings:
            return emoji_mappings[singular]

    return None

def check_shiny():
    """Check if this detection should be shiny (1/2500 chance)"""
    roll = random.randint(1, 2500)
    is_shiny = roll == 1
    return is_shiny, roll

def lookup_text_for_emojis(text: str) -> Dict[str, Any]:
    """Look up emojis for text with tokenization - returns same format as utils"""
    if not text or not text.strip():
        return {"mappings": {}, "found_emojis": []}
    
    try:
        # Tokenize text with MWE detection
        word_tokens = []
        for token in text.split():
            # Remove common punctuation
            token = token.strip('.,!?;:"()[]{}\'`')
            if token:
                word_tokens.append(token)
        
        tokens = emoji_tokenizer.tokenize(word_tokens)
        
        # Look up emojis for each token
        mappings = {}
        found_emojis = []
        
        for token in tokens:
            emoji = get_emoji_for_word(token)
            if emoji:
                mappings[token] = emoji
                if emoji not in found_emojis:
                    found_emojis.append(emoji)
        
        return {"mappings": mappings, "found_emojis": found_emojis}
        
    except Exception as e:
        logger.error(f"BLIP: Failed to process emoji mappings: {e}")
        return {"mappings": {}, "found_emojis": []}



# Device info logging (moved to analyzer)
logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
logger.info(f"CUDA device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
if torch.cuda.is_available():
    logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

def initialize_blip_analyzer() -> bool:
    """Initialize BLIP analyzer once at startup - fail fast"""
    global blip_analyzer
    try:
        logger.info("Initializing BLIP Analyzer...")
        
        blip_analyzer = BlipAnalyzer(
            image_size=IMAGE_SIZE,
            model_path="./model_base_capfilt_large.pth"
        )
        
        # Initialize the model
        if not blip_analyzer.initialize():
            logger.error("❌ Failed to initialize BLIP Analyzer")
            return False
            
        logger.info("✅ BLIP Analyzer initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error initializing BLIP Analyzer: {str(e)}")
        return False


def get_emojis_and_mappings_for_caption(caption: str) -> tuple[List[str], Dict[str, str]]:
    """Extract emojis and word mappings from caption using local emoji service (optimized - no HTTP requests)"""
    logger.debug(f"BLIP: get_emojis_and_mappings_for_caption called with: '{caption[:100]}...'")
    if not caption:
        return [], {}
    
    logger.debug("BLIP: Using local emoji service (no HTTP requests)...")
    
    try:
        # Use local emoji lookup instead of HTTP requests
        result = lookup_text_for_emojis(caption)
        
        emojis = result["found_emojis"]
        mappings = result["mappings"]
        
        logger.debug(f"BLIP: Found {len(mappings)} mappings: {mappings}")
        logger.debug(f"BLIP: Returning {len(emojis)} emojis: {emojis}")
        logger.debug(f"BLIP: Returning {len(mappings)} mappings: {mappings}")
        
        return emojis[:3], mappings  # Limit to 3 emojis as before
        
    except Exception as e:
        logger.error(f"BLIP: Failed to use local emoji service: {e}")
        
    # Return empty lists if service fails
    logger.debug("BLIP: No emojis found")
    return [], {}

def preprocess_image(image: Image.Image) -> Optional[torch.Tensor]:
    """Preprocess PIL Image for BLIP model"""
    try:
        logger.info("Preprocessing image from memory")
        
        # Ensure RGB format (handle different image modes)
        if image.mode != 'RGB':
            logger.info(f"Converting image from {image.mode} to RGB")
            image = image.convert('RGB')
        
        # Log original image dimensions
        logger.info(f"Original image size: {image.size}")
        
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        
        image_tensor = transform(image)
        
        # Log tensor shape before adding batch dimension
        logger.info(f"Image tensor shape after transform: {image_tensor.shape}")
        
        # Add batch dimension and move to device
        image_tensor = image_tensor.unsqueeze(0).to(device)
        
        # Convert to half precision if model is FP16
        if device.type == 'cuda' and hasattr(model, 'dtype') and model.dtype == torch.float16:
            image_tensor = image_tensor.half()
        
        # Log final tensor shape
        logger.info(f"Final image tensor shape: {image_tensor.shape}")
        
        return image_tensor
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
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


def process_image_for_caption(image: Image.Image) -> Dict[str, Any]:
    """
    Main processing function - takes PIL Image, returns caption data
    This is the core business logic, separated from HTTP concerns
    """
    try:
        # Generate caption using analyzer
        result = blip_analyzer.analyze_caption_from_array(image)
        
        if not result.get('success'):
            return {
                "success": False,
                "error": result.get('error', 'Caption generation failed')
            }
        
        # Get caption and emoji mappings
        caption = result.get('caption', '')
        emojis, word_mappings = get_emojis_and_mappings_for_caption(caption)
        
        return {
            "success": True,
            "caption": caption,
            "emojis": emojis,
            "word_mappings": word_mappings
        }
        
    except Exception as e:
        logger.error(f"Error processing image for caption: {e}")
        return {
            "success": False,
            "error": f"Processing failed: {str(e)}"
        }

def create_blip_response(caption: str, word_mappings: Dict[str, str], processing_time: float) -> Dict[str, Any]:
    """Create standardized BLIP response with metadata"""
    is_shiny, shiny_roll = check_shiny()
    
    # Convert word_mappings dict to emoji_mappings array format (like Ollama)
    emoji_mappings = []
    for word, emoji in word_mappings.items():
        emoji_mappings.append({
            "word": word,
            "emoji": emoji
        })
    
    prediction = {
        "text": caption,
        "emoji_mappings": emoji_mappings
    }
    
    # Add shiny flag for rare detections
    if is_shiny:
        prediction["shiny"] = True
        logger.info(f"✨ SHINY CAPTION GENERATED! Roll: {shiny_roll} ✨")
    
    return {
        "service": "blip",
        "status": "success",
        "predictions": [prediction],
        "metadata": {
            "processing_time": round(processing_time, 3),
            "model_info": {"framework": "BLIP (Bootstrapping Language-Image Pre-training)"}
        }
    }

def download_image_from_url(url: str) -> Image.Image:
    """Download image from URL and return as PIL Image"""
    try:
        headers = {'User-Agent': 'BLIP Caption Generation Service'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        if len(response.content) > MAX_FILE_SIZE:
            raise ValueError(f"Image too large. Max size: {MAX_FILE_SIZE/1024/1024}MB")
        
        # Return PIL Image directly from bytes
        from io import BytesIO
        image = Image.open(BytesIO(response.content))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        return image
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to download image: {str(e)}")

def validate_image_file(file_path: str) -> Image.Image:
    """Validate and load image file as PIL Image"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if not is_allowed_file(file_path):
        raise ValueError("File type not allowed")
    
    try:
        image = Image.open(file_path)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        return image
    except Exception as e:
        raise Exception(f"Failed to load image: {str(e)}")

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
print("BLIP service: CORS enabled for direct browser communication")

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
    try:
        model_status = "loaded" if blip_analyzer and blip_analyzer.model else "not_loaded"
        device_str = str(blip_analyzer.device) if blip_analyzer else "unknown"
        return jsonify({
            "status": "healthy",
            "model_status": model_status,
            "device": device_str
        })
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({"error": "Health check failed", "status": "error"}), 500

# V2 Compatibility Routes - Translate parameters and call V3
@app.route('/v2/analyze', methods=['GET'])
def analyze_v2_compat():
    """V2 compatibility - translate parameters to new analyze format"""
    import time
    from flask import request
    
    # Get V2 parameter
    image_url = request.args.get('image_url')
    
    if image_url:
        # Create new request args with new parameter name
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

@app.route('/v2/analyze_file', methods=['GET']) 
def analyze_file_v2_compat():
    """V2 file compatibility - translate parameters to new analyze format"""
    import time
    from flask import request
    
    # Get V2 parameter
    file_path = request.args.get('file_path')
    
    if file_path:
        # Create new request args with new parameter name
        new_args = {'file': file_path}
        
        # Create a mock request object for new analyze endpoint
        with app.test_request_context('/analyze', query_string=new_args):
            return analyze()
    else:
        # No parameters - let analyze handle the error
        with app.test_request_context('/analyze'):
            return analyze()

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

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    """Unified analyze endpoint - orchestrates input handling and processing"""
    import time
    from io import BytesIO
    start_time = time.time()
    
    def error_response(message: str, status_code: int = 400):
        return jsonify({
            "service": "blip",
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
            
            if url:
                try:
                    image = download_image_from_url(url)
                except Exception as e:
                    return error_response(str(e))
                
            else:  # file parameter
                try:
                    image = validate_image_file(file)
                except Exception as e:
                    return error_response(str(e))
        
        # Step 2: Process the image (unified processing path)
        processing_result = process_image_for_caption(image)
        
        # Step 3: Handle processing result
        if not processing_result["success"]:
            return error_response(processing_result["error"], 500)
        
        # Step 4: Create response
        response = create_blip_response(
            processing_result["caption"],
            processing_result["word_mappings"],
            time.time() - start_time
        )
        
        return jsonify(response)
        
    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        logger.error(f"Analyze API error: {e}")
        return error_response(f"Internal error: {str(e)}", 500)


if __name__ == '__main__':
    # Initialize model and emoji data
    logger.info("Starting BLIP service...")
    
    model_loaded = initialize_blip_analyzer()
    
    # Using local emoji file - no external dependencies
    
    if not model_loaded:
        logger.error("Failed to initialize BLIP analyzer. Service cannot function.")
        exit(1)
        
    
    # Determine host based on private mode
    host = "127.0.0.1" if PRIVATE else "0.0.0.0"
    
    logger.info(f"Starting BLIP service on {host}:{PORT}")
    logger.info(f"Private mode: {PRIVATE}")
    logger.info(f"Model loaded: {model_loaded}")
    logger.info("Emoji lookup: Local file mode")
    
    app.run(
        host=host,
        port=PORT,
        debug=False,
        use_reloader=False,
        threaded=True
    )
