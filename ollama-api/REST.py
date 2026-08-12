import asyncio
import base64
import json
import requests
import os
import sys
import uuid
import time
import logging
import random
import nltk
from nltk.tokenize import MWETokenizer
from io import BytesIO
from typing import Dict, Any, Optional, List


from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from ollama import AsyncClient
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Configuration Updates (GitHub-first pattern)
AUTO_UPDATE_STR = os.getenv('AUTO_UPDATE', 'true')
AUTO_UPDATE = AUTO_UPDATE_STR.lower() == 'true'

FOLDER = './uploads'
PRIVATE_STR = os.getenv('PRIVATE')
PORT_STR = os.getenv('PORT')
OLLAMA_HOST = os.getenv('OLLAMA_HOST')
VISION_MODEL = os.getenv('VISION_MODEL')
TEMPERATURE_STR = os.getenv('TEMPERATURE')
VISION_PROMPT = os.getenv('PROMPT')

# Validate critical configuration
if not PRIVATE_STR:
    raise ValueError("PRIVATE environment variable is required")
if not PORT_STR:
    raise ValueError("PORT environment variable is required")
if not OLLAMA_HOST:
    raise ValueError("OLLAMA_HOST environment variable is required")
if not VISION_MODEL:
    raise ValueError("VISION_MODEL environment variable is required")
if not TEMPERATURE_STR:
    raise ValueError("TEMPERATURE environment variable is required")
if not VISION_PROMPT:
    raise ValueError("PROMPT environment variable is required")

# Convert to appropriate types
PRIVATE = PRIVATE_STR.lower() in ['true', '1', 'yes']
PORT = int(PORT_STR)
TEMPERATURE = float(TEMPERATURE_STR)
MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', str(32 * 1024 * 1024)))  # 32MB default
MAX_RESPONSE_LENGTH = 4000  # Reasonable limit for responses
RAW_IMAGE_CONTENT_TYPES = {
    'image/jpeg',
    'image/png',
    'image/webp',
    'application/octet-stream',
}

# Ensure upload directory exists
os.makedirs(FOLDER, exist_ok=True)

# Common Ollama models - these will be dynamically fetched from Ollama
COMMON_MODELS = {
    "text": ["llama2", "llama2:13b", "mistral", "mixtral", "codellama", "dolphin-mistral", "phi"],
    "vision": ["llava", "llava:13b", "bakllava"],
    "code": ["codellama", "codellama:13b", "deepseek-coder:6.7b"]
}

# Load emoji mappings from central API
emoji_mappings = {}
emoji_tokenizer = None


def is_raw_image_request() -> bool:
    return (request.content_type or '').split(';', 1)[0].strip().lower() in RAW_IMAGE_CONTENT_TYPES

def load_emoji_mappings():
    """Load emoji mappings from GitHub, fall back to local cache"""
    github_url = "https://raw.githubusercontent.com/ice9innovations/animal-farm/refs/heads/main/config/emoji_mappings.json"
    local_cache_path = 'emoji_mappings.json'

    if AUTO_UPDATE:
        try:
            logger.info(f"🔄 LLaMa: Loading emoji mappings from GitHub: {github_url}")
            response = requests.get(github_url, timeout=10.0)
            response.raise_for_status()
            
            # Save to local cache (preserve emoji characters)
            with open(local_cache_path, 'w', encoding='utf-8') as f:
                json.dump(response.json(), f, indent=2, ensure_ascii=False)
            
            logger.info("✅ LLaMa: Loaded emoji mappings from GitHub and cached locally")
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️  LLaMa: Failed to load emoji mappings from GitHub ({e}), falling back to local cache")
    
    # Fall back to local cache
    try:
        with open(local_cache_path, 'r') as f:
            local_mappings = json.load(f)
            logger.info("✅ LLaMa: Successfully loaded emoji mappings from local cache")
            return local_mappings
    except Exception as local_error:
        logger.error(f"❌ LLaMa: Failed to load local emoji mappings: {local_error}")
        raise Exception(f"Both GitHub and local emoji mappings failed: GitHub download disabled or failed, Local cache={local_error}")

def load_mwe_mappings():
    """Load MWE mappings from GitHub, fall back to local cache, convert to tuples"""
    github_url = "https://raw.githubusercontent.com/ice9innovations/animal-farm/refs/heads/main/config/mwe.txt"
    local_cache_path = 'mwe.txt'
    
    def process_mwe_lines(mwe_text):
        """Convert MWE text lines to tuples for MWETokenizer"""
        mwe_tuples = []
        for line in mwe_text:
            if line.strip():  # Skip empty lines
                # Convert underscore format to word tuples (e.g., "street_sign" -> ("street", "sign"))
                mwe_tuples.append(tuple(line.strip().replace('_', ' ').split()))
        return mwe_tuples
    
    if AUTO_UPDATE:
        try:
            logger.info(f"🔄 LLaMa: Loading MWE mappings from GitHub: {github_url}")
            response = requests.get(github_url, timeout=10.0)
            response.raise_for_status()
            
            # Save to local cache
            with open(local_cache_path, 'w') as f:
                f.write(response.text)
            
            mwe_text = response.text.splitlines()
            mwe_tuples = process_mwe_lines(mwe_text)
            logger.info(f"✅ LLaMa: Loaded MWE mappings from GitHub and cached locally")
            return mwe_tuples
        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️  LLaMa: Failed to load MWE mappings from GitHub ({e}), falling back to local cache")
    
    # Fall back to local cache
    try:
        with open(local_cache_path, 'r') as f:
            local_mwe_text = f.read().splitlines()
            mwe_tuples = process_mwe_lines(local_mwe_text)
            logger.info(f"✅ LLaMa: Successfully loaded {len(mwe_tuples)} multi-word expressions from local cache")
            return mwe_tuples
    except Exception as local_error:
        logger.error(f"❌ LLaMa: Failed to load local MWE mappings: {local_error}")
        raise Exception(f"Both GitHub and local MWE mappings failed: GitHub download disabled or failed, Local cache={local_error}")

def ensure_ollama_models():
    """Download required Ollama vision model if it doesn't exist"""
    import subprocess
    
    if VISION_MODEL:
        try:
            # Check if model exists
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
            if VISION_MODEL not in result.stdout:
                logger.info(f"🔄 LLaMa: Downloading vision model: {VISION_MODEL}")
                subprocess.run(['ollama', 'pull', VISION_MODEL], check=True)
                logger.info(f"✅ LLaMa: Successfully downloaded vision model: {VISION_MODEL}")
            else:
                logger.info(f"✅ LLaMa: Vision model {VISION_MODEL} already available")
        except Exception as e:
            logger.error(f"❌ LLaMa: Failed to download vision model {VISION_MODEL}: {e}")

# Load emoji mappings on startup
emoji_mappings = load_emoji_mappings()
mwe_mappings = load_mwe_mappings()

# Initialize MWE tokenizer with the loaded mappings (already converted to tuples)
emoji_tokenizer = MWETokenizer(mwe_mappings, separator='_')

# Ensure Ollama models are downloaded
ensure_ollama_models()

# Priority overrides for critical ambiguities - defined once at module level
PRIORITY_OVERRIDES = {
    'glass': '🥛',        # drinking glass, not eyewear
    'glasses': '👓',      # eyewear (explicit to avoid fallback)
    'wood': '🌲',         # base material form
    'wooden': '🌲',       # morphological variant
    'metal': '🔧',        # base material form  
    'metallic': '🔧',     # morphological variant
}

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
        # Split and clean punctuation from each word
        word_tokens = []
        for token in text.split():
            if token:
                # Strip punctuation from each word
                clean_token = token.strip('.,!?;:"()[]{}\'`')
                if clean_token:
                    word_tokens.append(clean_token.lower())
        
        tokens = emoji_tokenizer.tokenize(word_tokens)

        # Debug logging
        logger.debug(f"LLaMa: Input tokens: {word_tokens}")
        logger.debug(f"LLaMa: MWE tokens: {tokens}")

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
        logger.error(f"LLaMa: Failed to process emoji mappings: {e}")
        return {"mappings": {}, "found_emojis": []}

def get_emojis_for_text(text: str) -> List[Dict[str, str]]:
    """Extract emojis from sentence using local emoji service"""
    logger.debug(f"LLaMa: get_emojis_for_text called with: '{text[:100]}...'")
    if not text or not text.strip():
        return []
    
    try:
        # Use local emoji lookup instead of utils package
        result = lookup_text_for_emojis(text)
        
        # Convert to expected format with deduplication
        found_mappings = []
        seen_words = set()
        for word, emoji in result["mappings"].items():
            # Normalize word format: lowercase with underscores
            normalized_word = word.lower().replace(' ', '_')
            # Only add if we haven't seen this normalized word before
            if normalized_word not in seen_words:
                seen_words.add(normalized_word)
                
                is_shiny, shiny_roll = check_shiny()
                
                mapping = {"word": normalized_word, "emoji": emoji}
                
                # Add shiny flag only for shiny detections
                if is_shiny:
                    mapping["shiny"] = True
                    logger.info(f"✨ SHINY {normalized_word.upper()} EMOJI DETECTED! Roll: {shiny_roll} ✨")
                
                found_mappings.append(mapping)
        
        logger.debug(f"LLaMa: Found {len(found_mappings)} emoji mappings")
        return found_mappings
        
    except Exception as e:
        logger.error(f"LLaMa: Failed to process emoji mappings: {e}")
        return []

def create_ollama_response(data: Dict[str, Any], processing_time: float) -> Dict[str, Any]:
    """Create standardized Ollama response with emoji mappings"""
    text_response = data.get('response', '')
    emoji_mappings = data.get('emoji_mappings', [])

    is_shiny, shiny_roll = check_shiny()
    
    prediction = {
        "text": text_response
    }
    
    # Add emoji mappings if found
    if emoji_mappings:
        prediction["emoji_mappings"] = emoji_mappings
    
    # Add shiny flag only for shiny detections
    if is_shiny:
        prediction["shiny"] = True
        logger.info(f"✨ SHINY OLLAMA VISION ANALYSIS! Roll: {shiny_roll} ✨")
    
    return {
        "service": "ollama",
        "status": "success",
        "predictions": [prediction],
        "metadata": {
            "processing_time": round(processing_time, 3),
            "model_info": {
                "framework": "Ollama",
                "model": data.get('model_used', VISION_MODEL),
                "prompt": data.get('prompt', VISION_PROMPT)
            }
        }
    }

# Removed context lookup function - using simple local lookup now

async def get_available_models() -> Dict[str, List[str]]:
    """Get available models from Ollama server"""
    try:
        client = AsyncClient(host=OLLAMA_HOST)
        models_response = await client.list()
        available_models = [model.model for model in models_response.models]
        
        # Categorize models
        categorized = {
            "text": [m for m in available_models if any(text_model in m for text_model in COMMON_MODELS["text"])],
            "vision": [m for m in available_models if any(vision_model in m for vision_model in COMMON_MODELS["vision"])],
            "code": [m for m in available_models if any(code_model in m for code_model in COMMON_MODELS["code"])],
            "other": [m for m in available_models if not any(
                any(known_model in m for known_model in category_models) 
                for category_models in COMMON_MODELS.values()
            )]
        }
        
        return categorized
    except Exception as e:
        print(f"Failed to get models from Ollama: {e}")
        return COMMON_MODELS


def process_image_for_ollama(image: Image.Image, prompt: str = None, model: str = None, temperature: float = None) -> Dict[str, Any]:
    """
    Main processing function - takes PIL Image, returns Ollama vision analysis data
    This is the core business logic, separated from HTTP concerns
    Uses pure in-memory processing with direct PIL Image support
    """
    start_time = time.time()
    model = model or VISION_MODEL
    prompt = prompt or VISION_PROMPT
    temperature = temperature if temperature is not None else TEMPERATURE
    
    try:
        # Ensure image is in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert PIL Image to base64 for Ollama (in memory)
        from io import BytesIO
        jpg_buffer = BytesIO()
        image.save(jpg_buffer, 'JPEG', quality=95)
        jpg_data = jpg_buffer.getvalue()
        jpg_buffer.close()
        
        # Encode for Ollama
        image_data = base64.b64encode(jpg_data).decode('utf-8')
        
        # Process with Ollama (async in sync wrapper)
        import asyncio
        client = AsyncClient(host=OLLAMA_HOST)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        ollama_response = loop.run_until_complete(client.generate(
            model=model,
            prompt=prompt,
            images=[image_data],
            stream=False,
            options={"temperature": temperature}
        ))
        
        loop.close()
        
        processing_time = round(time.time() - start_time, 3)
        output = ollama_response["response"]
        
        # Truncate if too long
        if len(output) > MAX_RESPONSE_LENGTH:
            output = output[:MAX_RESPONSE_LENGTH] + "... [truncated]"
        
        # Extract emojis from response
        emoji_list = get_emojis_for_text(output)

        return {
            "success": True,
            "data": {
                "response": output,
                "model_used": model,
                "prompt": prompt,
                "response_length": len(output),
                "emoji_mappings": emoji_list,
                "type": "vision",
                "temperature": temperature
            },
            "processing_time": processing_time
        }
        
    except Exception as e:
        processing_time = round(time.time() - start_time, 3)
        return {
            "success": False,
            "error": f"Image analysis failed: {str(e)}",
            "processing_time": processing_time
        }



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

# Enable CORS for direct browser access (eliminates PHP proxy)
CORS(app, origins=["*"], methods=["GET", "POST", "OPTIONS"])
print("Ollama service: CORS enabled for direct browser communication")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    # Test Ollama connectivity with actual model test
    try:
        # Test with a simple text generation to ensure Ollama is actually working
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Try to get models first
        available_models = loop.run_until_complete(get_available_models())
        
        # Test actual text generation capability
        client = AsyncClient(host=OLLAMA_HOST)
        test_response = loop.run_until_complete(client.chat(
            model=VISION_MODEL,
            messages=[{'role': 'user', 'content': 'Hi'}]
        ))
        
        loop.close()
        
        ollama_status = "connected"
        status = "healthy"
        
    except Exception as e:
        ollama_status = f"error: {str(e)}"
        status = "unhealthy"
        available_models = {}
        
        return jsonify({
            "status": status,
            "reason": f"Ollama service error: {str(e)}",
            "service": "Ollama LLM Integration"
        }), 503
    
    return jsonify({
        "status": status,
        "service": "Ollama LLM Integration",
        "ollama": {
            "host": OLLAMA_HOST,
            "status": ollama_status,
            "default_vision_model": VISION_MODEL
        },
        "available_models": available_models,
        "features": {
            "text_generation": True,
            "image_analysis": True,
            "model_selection": True,
            "streaming": False  # Not implemented in this version
        },
        "endpoints": [
            "GET /health - Health check",
            "GET /models - List available models",
            "GET,POST /analyze - Unified endpoint (URL/file/upload)",
            "GET /v3/analyze - V3 compatibility",
            "POST /image - Analyze image with text"
        ]
    })

@app.route('/models', methods=['GET'])
def list_models():
    """List available Ollama models"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        models = loop.run_until_complete(get_available_models())
        loop.close()
        
        return jsonify({
            "models": models,
            "defaults": {
                "vision": VISION_MODEL
            },
            "status": "success"
        })
    except Exception as e:
        return jsonify({
            "error": f"Failed to fetch models: {str(e)}",
            "status": "error"
        }), 500

# V2 Compatibility Routes - Translate parameters and call V3
@app.route('/v2/analyze', methods=['GET'])
def analyze_v2_compat():
    """V2 compatibility - translate parameters to new analyze format"""
    image_url = request.args.get('image_url')
    
    if image_url:
        # Parameter translation: image_url -> url
        new_args = {'url': image_url}
        # Copy other parameters
        for key, value in request.args.items():
            if key != 'image_url':
                new_args[key] = value
        
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
        # Copy other parameters
        for key, value in request.args.items():
            if key != 'file_path':
                new_args[key] = value
        
        with app.test_request_context('/analyze', query_string=new_args):
            return analyze()
    else:
        with app.test_request_context('/analyze'):
            return analyze()

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    """Unified analyze endpoint - orchestrates input handling and processing"""
    start_time = time.time()
    
    def error_response(message: str, status_code: int = 400):
        return jsonify({
            "service": "ollama",
            "status": "error",
            "predictions": [],
            "error": {"message": message},
            "metadata": {"processing_time": round(time.time() - start_time, 3)}
        }), status_code
    
    try:
        # Get optional parameters
        prompt = request.args.get('prompt') or request.form.get('prompt') or VISION_PROMPT
        model = request.args.get('model') or request.form.get('model') or VISION_MODEL
        temperature_param = request.args.get('temperature') or request.form.get('temperature')
        temperature = float(temperature_param) if temperature_param else TEMPERATURE
        
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
                    response = requests.get(url, timeout=15)
                    response.raise_for_status()
                    
                    if len(response.content) > MAX_FILE_SIZE:
                        return error_response("Downloaded file too large")
                    
                    from io import BytesIO
                    image = Image.open(BytesIO(response.content)).convert('RGB')
                    
                except Exception as e:
                    return error_response(f"Failed to download/process image: {str(e)}")
            else:  # file_path
                # Load file directly into memory
                if not os.path.exists(file_path):
                    return error_response(f"File not found: {file_path}")
                
                try:
                    image = Image.open(file_path).convert('RGB')
                except Exception as e:
                    return error_response(f"Failed to load image file: {str(e)}", 500)
        
        # Step 2: Process the image (unified processing path)
        processing_result = process_image_for_ollama(image, prompt, model, temperature)
        
        # Step 3: Handle processing result
        if not processing_result["success"]:
            return error_response(processing_result["error"], 500)
        
        # Step 4: Create response
        response = create_ollama_response(
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
    """V3 compatibility - calls new analyze function directly"""
    return analyze()



@app.route('/image', methods=['POST'])
def analyze_image_with_prompt():
    """Analyze uploaded image with custom prompt"""
    try:
        # Get prompt from form data or JSON (handle different content types)
        prompt = request.args.get('prompt') or request.headers.get('X-Ice9-Prompt')
        model = request.args.get('model') or request.headers.get('X-Ice9-Model')

        if not is_raw_image_request():
            prompt = prompt or request.form.get('prompt')
            model = model or request.form.get('model')
        
        # Fallback to JSON if form data is empty and content-type is JSON
        if not prompt and request.is_json:
            json_data = request.get_json(silent=True)
            if json_data:
                prompt = json_data.get('prompt')
                model = model or json_data.get('model')
        
        # Set defaults
        prompt = prompt or VISION_PROMPT
        model = model or VISION_MODEL

        if is_raw_image_request():
            file_data = request.get_data(cache=False)
            if not file_data:
                return jsonify({
                    "error": "No image body provided",
                    "status": "error"
                }), 400
            if len(file_data) > MAX_FILE_SIZE:
                return jsonify({
                    "error": f"File too large. Maximum size: {MAX_FILE_SIZE // 1024 // 1024}MB",
                    "status": "error"
                }), 400
            try:
                image = Image.open(BytesIO(file_data)).convert('RGB')
                result = process_image_for_ollama(image, prompt, model)
                if not result.get("success"):
                    return jsonify({
                        "error": result.get("error", "Image analysis failed"),
                        "status": "error"
                    }), 500
                return jsonify(result)
            except Exception as e:
                return jsonify({
                    "error": f"Image analysis failed: {str(e)}",
                    "status": "error"
                }), 500
        
        if not request.files:
            return jsonify({
                "error": "No image file uploaded",
                "status": "error"
            }), 400
        
        for field_name, file_data in request.files.items():
            if not file_data.filename:
                continue
                
            try:
                image_bytes = file_data.read()
                if len(image_bytes) > MAX_FILE_SIZE:
                    return jsonify({
                        "error": f"File too large. Maximum size: {MAX_FILE_SIZE // 1024 // 1024}MB",
                        "status": "error"
                    }), 400

                image = Image.open(BytesIO(image_bytes)).convert('RGB')
                result = process_image_for_ollama(image, prompt, model)
                if not result.get("success"):
                    return jsonify({
                        "error": result.get("error", "Image analysis failed"),
                        "status": "error"
                    }), 500
                return jsonify(result)
                
            except Exception as e:
                return jsonify({
                    "error": f"Image analysis failed: {str(e)}",
                    "status": "error"
                }), 500
        
        return jsonify({
            "error": "No valid image file found",
            "status": "error"
        }), 400
        
    except Exception as e:
        return jsonify({
            "error": f"Image analysis failed: {str(e)}",
            "status": "error"
        }), 500


if __name__ == '__main__':
    # Initialize services
    logger.info("Starting Ollama LLM service...")
    logger.info("MAIN BLOCK REACHED - starting initialization")
    
    
    # Using local emoji file - no external dependencies
    
    # Test Ollama connectivity on startup
    logger.info(f"Starting Ollama LLM API on port {PORT}")
    logger.info(f"Ollama host: {OLLAMA_HOST}")
    logger.info(f"Default vision model: {VISION_MODEL}")
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        models = loop.run_until_complete(get_available_models())
        loop.close()
        logger.info(f"Available models: {sum(len(v) for v in models.values())} total")
    except Exception as e:
        logger.warning(f"Could not connect to Ollama: {e}")
        logger.warning("Make sure Ollama is running and accessible")
    
    host = "0.0.0.0" if not PRIVATE else "127.0.0.1"
    logger.info(f"Starting on {host}:{PORT}")
    logger.info(f"Private mode: {PRIVATE}")
    logger.info("Emoji lookup: Local file mode")
    
    app.run(host=host, port=int(PORT), debug=False, threaded=True)
