import os
import sys
import time
import logging
import requests
import torch
import clip
from io import BytesIO
from typing import Optional

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

PORT_STR = os.getenv('PORT')
PRIVATE_STR = os.getenv('PRIVATE')
CLIP_MODEL = os.getenv('CLIP_MODEL')

if not PORT_STR:
    raise ValueError("PORT environment variable is required")
if not PRIVATE_STR:
    raise ValueError("PRIVATE environment variable is required")
if not CLIP_MODEL:
    raise ValueError("CLIP_MODEL environment variable is required")

PORT = int(PORT_STR)
PRIVATE = PRIVATE_STR.lower() in ['true', '1', 'yes']
MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', str(32 * 1024 * 1024)))  # 32MB default
RAW_IMAGE_CONTENT_TYPES = {
    'image/jpeg',
    'image/png',
    'image/webp',
    'application/octet-stream',
}

device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    device = "mps"

logger.info(f"Loading CLIP model {CLIP_MODEL} on {device}...")
model, preprocess = clip.load(CLIP_MODEL, device=device)

if device == "cuda":
    model = model.half()
    logger.info("Applied FP16 — 50% VRAM reduction")

logger.info(f"CLIP model {CLIP_MODEL} ready")


def is_raw_image_request() -> bool:
    return (request.content_type or '').split(';', 1)[0].strip().lower() in RAW_IMAGE_CONTENT_TYPES


def encode_image_only(image: Image.Image) -> Optional[list]:
    """Return a normalized CLIP image embedding as a plain Python list.

    Suitable for pgvector storage. Returns None on failure.
    """
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')

        image_tensor = preprocess(image).unsqueeze(0).to(device)
        if device == "cuda" and model.dtype == torch.float16:
            image_tensor = image_tensor.half()

        with torch.no_grad():
            if device == "cuda" and model.dtype == torch.float16:
                with torch.amp.autocast('cuda'):
                    image_features = model.encode_image(image_tensor)
            else:
                image_features = model.encode_image(image_tensor)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_embedding = image_features.squeeze().float().cpu().tolist()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return image_embedding

    except Exception as e:
        logger.error(f"encode_image_only failed: {e}")
        return None


def score_caption(image: Image.Image, caption: str) -> Optional[float]:
    """Compute cosine similarity between a PIL Image and a caption string.

    Returns the similarity score as a float, or None on failure.
    """
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')

        image_tensor = preprocess(image).unsqueeze(0).to(device)
        if device == "cuda" and model.dtype == torch.float16:
            image_tensor = image_tensor.half()

        text_tokens = clip.tokenize([caption], truncate=True).to(device)

        with torch.no_grad():
            if device == "cuda" and model.dtype == torch.float16:
                with torch.amp.autocast('cuda'):
                    image_features = model.encode_image(image_tensor)
                    text_features = model.encode_text(text_tokens)
            else:
                image_features = model.encode_image(image_tensor)
                text_features = model.encode_text(text_tokens)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            similarity = (image_features @ text_features.T).item()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return similarity

    except Exception as e:
        logger.error(f"score_caption failed: {e}")
        return None


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

CORS(app, origins=["*"], methods=["GET", "POST", "OPTIONS"])


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "clip-score",
        "model": CLIP_MODEL,
        "device": device,
        "endpoints": [
            "GET /health",
            "GET,POST /score - caption + (url | file | multipart)",
            "POST /embed/image - image embedding only (url | file | multipart)",
            "POST /embed/text - batch text-only embeddings",
            "(deprecated) /v3/score, /v3/embed/text",
        ]
    })


@app.route('/embed/text', methods=['POST'])
@app.route('/v3/embed/text', methods=['POST'])
def embed_text():
    """Return CLIP text embeddings for a batch of terms.

    Accepts JSON: {"terms": ["dog", "cat", "truck"]}
    Returns:      {"embeddings": {"dog": [...], "cat": [...], "truck": [...]}}

    No image required — runs the text encoder only. Embeddings are
    L2-normalized, identical to the text_embedding returned by /v3/score.
    """
    start_time = time.time()

    def error_response(message: str, status_code: int = 400):
        return jsonify({
            "service": "clip-score",
            "status": "error",
            "error": {"message": message},
            "metadata": {"processing_time": round(time.time() - start_time, 3)},
        }), status_code

    try:
        if not request.is_json:
            return error_response("Request must be JSON with a 'terms' list")

        data = request.get_json()
        terms = data.get('terms', [])

        if not terms or not isinstance(terms, list):
            return error_response("Must provide a non-empty 'terms' list")
        if len(terms) > 500:
            return error_response("Too many terms (max 500)")

        # Deduplicate, strip whitespace, drop empties — preserve first-seen order
        seen = {}
        for t in terms:
            t = t.strip()
            if t and t not in seen:
                seen[t] = None
        unique_terms = list(seen.keys())

        if not unique_terms:
            return error_response("No valid terms provided")

        text_tokens = clip.tokenize(unique_terms, truncate=True).to(device)

        with torch.no_grad():
            if device == "cuda" and model.dtype == torch.float16:
                with torch.amp.autocast('cuda'):
                    text_features = model.encode_text(text_tokens)
            else:
                text_features = model.encode_text(text_tokens)

            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            embeddings_list = text_features.float().cpu().tolist()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        embeddings = {term: emb for term, emb in zip(unique_terms, embeddings_list)}

        return jsonify({
            "service": "clip-score",
            "status": "success",
            "embeddings": embeddings,
            "metadata": {
                "processing_time": round(time.time() - start_time, 3),
                "term_count": len(embeddings),
                "model_info": {
                    "framework": "openai-clip",
                    "model": CLIP_MODEL,
                    "device": device,
                },
            },
        })

    except Exception as e:
        logger.error(f"embed_text endpoint error: {e}")
        return error_response(f"Internal error: {str(e)}", 500)


@app.route('/embed/image', methods=['POST'])
def embeddings():
    start_time = time.time()

    def error_response(message: str, status_code: int = 400):
        return jsonify({
            "service": "clip-score",
            "status": "error",
            "error": {"message": message},
            "metadata": {"processing_time": round(time.time() - start_time, 3)}
        }), status_code

    try:
        image = None

        if is_raw_image_request():
            file_data = request.get_data(cache=False)
            if not file_data:
                return error_response("No image body provided")
            if len(file_data) > MAX_FILE_SIZE:
                return error_response(f"File too large (max {MAX_FILE_SIZE // 1024 // 1024}MB)")
            try:
                image = Image.open(BytesIO(file_data)).convert('RGB')
            except Exception as e:
                return error_response(f"Failed to open raw image body: {e}", 500)

        elif 'file' in request.files:
            uploaded_file = request.files['file']
            if uploaded_file.filename == '':
                return error_response("No file selected")
            uploaded_file.seek(0, 2)
            if uploaded_file.tell() > MAX_FILE_SIZE:
                return error_response(f"File too large (max {MAX_FILE_SIZE // 1024 // 1024}MB)")
            uploaded_file.seek(0)
            try:
                image = Image.open(BytesIO(uploaded_file.read())).convert('RGB')
            except Exception as e:
                return error_response(f"Failed to open uploaded image: {e}", 500)

        else:
            if request.is_json:
                data = request.get_json()
                url = data.get('url')
                file_path = data.get('file')
            else:
                url = request.form.get('url') or request.args.get('url')
                file_path = request.form.get('file') or request.args.get('file')

            if not url and not file_path:
                return error_response("Must provide image via multipart upload, 'url', or 'file' parameter")
            if url and file_path:
                return error_response("Cannot provide both 'url' and 'file' parameters")

            if url:
                try:
                    r = requests.get(url, timeout=15)
                    r.raise_for_status()
                    if len(r.content) > MAX_FILE_SIZE:
                        return error_response(f"Downloaded image too large (max {MAX_FILE_SIZE // 1024 // 1024}MB)")
                    image = Image.open(BytesIO(r.content)).convert('RGB')
                except Exception as e:
                    return error_response(f"Failed to download image: {e}")
            else:
                if not os.path.exists(file_path):
                    return error_response(f"File not found: {file_path}")
                try:
                    image = Image.open(file_path).convert('RGB')
                except Exception as e:
                    return error_response(f"Failed to open image file: {e}", 500)

        image_embedding = encode_image_only(image)

        if image_embedding is None:
            return error_response("Failed to compute image embedding", 500)

        return jsonify({
            "service": "clip-score",
            "status": "success",
            "image_embedding": image_embedding,
            "metadata": {
                "processing_time": round(time.time() - start_time, 3),
                "model_info": {
                    "framework": "openai-clip",
                    "model": CLIP_MODEL,
                    "device": device,
                }
            }
        })

    except Exception as e:
        logger.error(f"embeddings endpoint error: {e}")
        return error_response(f"Internal error: {str(e)}", 500)


@app.route('/score', methods=['GET', 'POST'])
@app.route('/v3/score', methods=['GET', 'POST'])
def score():
    start_time = time.time()

    def error_response(message: str, status_code: int = 400):
        return jsonify({
            "service": "clip-score",
            "status": "error",
            "similarity_score": None,
            "error": {"message": message},
            "metadata": {"processing_time": round(time.time() - start_time, 3)}
        }), status_code

    try:
        # --- get caption ---
        if request.is_json:
            data = request.get_json()
            caption = data.get('caption')
        else:
            caption = request.form.get('caption') or request.args.get('caption')

        if not caption or not caption.strip():
            return error_response("Must provide non-empty 'caption' parameter")
        caption = caption.strip()

        # --- get image ---
        image = None

        if request.method == 'POST' and is_raw_image_request():
            file_data = request.get_data(cache=False)
            if not file_data:
                return error_response("No image body provided")
            if len(file_data) > MAX_FILE_SIZE:
                return error_response(f"File too large (max {MAX_FILE_SIZE // 1024 // 1024}MB)")
            try:
                image = Image.open(BytesIO(file_data)).convert('RGB')
            except Exception as e:
                return error_response(f"Failed to open raw image body: {e}", 500)

        elif request.method == 'POST' and 'file' in request.files:
            uploaded_file = request.files['file']
            if uploaded_file.filename == '':
                return error_response("No file selected")
            uploaded_file.seek(0, 2)
            if uploaded_file.tell() > MAX_FILE_SIZE:
                return error_response(f"File too large (max {MAX_FILE_SIZE // 1024 // 1024}MB)")
            uploaded_file.seek(0)
            try:
                image = Image.open(BytesIO(uploaded_file.read())).convert('RGB')
            except Exception as e:
                return error_response(f"Failed to open uploaded image: {e}", 500)

        else:
            if request.is_json:
                data = request.get_json()
                url = data.get('url')
                file_path = data.get('file')
            else:
                url = request.form.get('url') or request.args.get('url')
                file_path = request.form.get('file') or request.args.get('file')

            if not url and not file_path:
                return error_response("Must provide image via multipart upload, 'url', or 'file' parameter")
            if url and file_path:
                return error_response("Cannot provide both 'url' and 'file' parameters")

            if url:
                try:
                    r = requests.get(url, timeout=15)
                    r.raise_for_status()
                    if len(r.content) > MAX_FILE_SIZE:
                        return error_response(f"Downloaded image too large (max {MAX_FILE_SIZE // 1024 // 1024}MB)")
                    image = Image.open(BytesIO(r.content)).convert('RGB')
                except Exception as e:
                    return error_response(f"Failed to download image: {e}")
            else:
                if not os.path.exists(file_path):
                    return error_response(f"File not found: {file_path}")
                try:
                    image = Image.open(file_path).convert('RGB')
                except Exception as e:
                    return error_response(f"Failed to open image file: {e}", 500)

        similarity = score_caption(image, caption)

        if similarity is None:
            return error_response("Failed to compute similarity score", 500)

        return jsonify({
            "service": "clip-score",
            "status": "success",
            "similarity_score": round(float(similarity), 4),
            "caption": caption,
            "metadata": {
                "processing_time": round(time.time() - start_time, 3),
                "model_info": {
                    "framework": "openai-clip",
                    "model": CLIP_MODEL,
                    "device": device,
                }
            }
        })

    except Exception as e:
        logger.error(f"score endpoint error: {e}")
        return error_response(f"Internal error: {str(e)}", 500)


if __name__ == '__main__':
    logger.info(f"Starting clip-score service on port {PORT}")
    logger.info(f"Model: {CLIP_MODEL} | Device: {device} | Private: {PRIVATE}")
    host = "0.0.0.0" if not PRIVATE else "127.0.0.1"
    app.run(host=host, port=PORT, debug=False, threaded=True)
