import os
import logging
import tempfile
import time
import requests
from contextlib import contextmanager
from io import BytesIO
from typing import Optional, Dict, Any, Tuple

from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from speciesnet import SpeciesNet, DEFAULT_MODEL

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB - camera trap images can be large
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff', 'webp'}
RAW_IMAGE_CONTENT_TYPES = {
    'image/jpeg',
    'image/png',
    'image/webp',
    'image/tiff',
    'application/octet-stream',
}

PRIVATE_STR = os.getenv('PRIVATE')
PORT_STR = os.getenv('PORT')
MODEL_NAME = os.getenv('MODEL', DEFAULT_MODEL)

# Validate critical configuration
if not PRIVATE_STR:
    raise ValueError("PRIVATE environment variable is required")
if not PORT_STR:
    raise ValueError("PORT environment variable is required")

PRIVATE = PRIVATE_STR.lower() == 'true'
PORT = int(PORT_STR)
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.5'))

# Global SpeciesNet model - initialize once at startup
speciesnet_model = None


def is_raw_image_request() -> bool:
    return (request.content_type or '').split(';', 1)[0].strip().lower() in RAW_IMAGE_CONTENT_TYPES

# Taxonomy → group mappings loaded from family_groups.csv
# The same CSV feeds three dicts keyed by family, order, and class names.
# Known family/order/class names don't collide so we can tell them apart by
# which taxonomy field they appear in.
FAMILY_GROUPS: Dict[str, str] = {}
ORDER_GROUPS: Dict[str, str] = {}
CLASS_GROUPS: Dict[str, str] = {}

# Known taxonomy classes for disambiguation when loading the CSV
_CLASSES = {'mammalia', 'aves', 'reptilia', 'amphibia', 'arachnida'}
_ORDERS = {
    'squamata', 'testudines', 'crocodilia', 'crocodylia', 'anura', 'caudata',
    'rodentia', 'carnivora', 'artiodactyla', 'perissodactyla', 'primates',
    'chiroptera', 'lagomorpha', 'eulipotyphla', 'afrosoricida', 'proboscidea',
    'hyracoidea', 'tubulidentata', 'macroscelidea', 'pholidota', 'pilosa',
    'cingulata', 'dasyuromorphia', 'diprotodontia', 'peramelemorphia',
    'didelphimorphia', 'monotremata', 'accipitriformes', 'falconiformes',
    'strigiformes', 'galliformes', 'anseriformes', 'passeriformes',
    'psittaciformes', 'piciformes', 'columbiformes', 'gruiformes',
    'charadriiformes', 'pelecaniformes', 'suliformes', 'araneae',
}


def load_family_groups() -> None:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'family_groups.csv')
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(',', 1)
                if len(parts) != 2:
                    continue
                key, value = parts[0].strip(), parts[1].strip()
                if key in _CLASSES:
                    CLASS_GROUPS[key] = value
                elif key in _ORDERS:
                    ORDER_GROUPS[key] = value
                else:
                    FAMILY_GROUPS[key] = value
        logger.info(
            f"Loaded {len(FAMILY_GROUPS)} family, {len(ORDER_GROUPS)} order, "
            f"{len(CLASS_GROUPS)} class group mappings"
        )
    except FileNotFoundError:
        logger.warning(f"family_groups.csv not found at {path} - group field will be omitted")


def initialize_speciesnet() -> bool:
    """Initialize SpeciesNet model once at startup - fail fast."""
    global speciesnet_model
    try:
        logger.info(f"Initializing SpeciesNet model: {MODEL_NAME}")
        speciesnet_model = SpeciesNet(MODEL_NAME)
        logger.info("SpeciesNet model initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize SpeciesNet: {e}")
        return False


def is_allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def ext_from_content_type(content_type: str, fallback_url: str = '') -> str:
    """Guess a file extension from a Content-Type header, with URL fallback."""
    ct = content_type.lower()
    if 'jpeg' in ct or 'jpg' in ct:
        return '.jpg'
    if 'png' in ct:
        return '.png'
    if 'webp' in ct:
        return '.webp'
    if 'tiff' in ct or 'tif' in ct:
        return '.tiff'
    # Fall back to the URL's own extension
    parts = fallback_url.split('?')[0].rsplit('.', 1)
    return f'.{parts[1].lower()}' if len(parts) > 1 else '.jpg'


# /dev/shm is a RAM-backed tmpfs on Linux - files here never touch disk
_RAM_DIR = '/dev/shm' if os.path.isdir('/dev/shm') else None


@contextmanager
def temp_image_file(data: bytes, ext: str):
    """Write bytes to a RAM-backed temp file, yield its path, delete on exit.

    Uses /dev/shm (tmpfs) so image data never touches disk.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=ext, dir=_RAM_DIR, delete=False)
    try:
        tmp.write(data)
        tmp.close()
        yield tmp.name
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


def image_size_from_bytes(data: bytes) -> Tuple[int, int]:
    """Return (width, height) from raw image bytes."""
    img = Image.open(BytesIO(data))
    return img.size  # (width, height)


def image_size_from_path(path: str) -> Tuple[int, int]:
    """Return (width, height) from an image file on disk."""
    img = Image.open(path)
    return img.size  # (width, height)


def run_prediction(
    filepath: str,
    country: Optional[str] = None,
    admin1_region: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
) -> Dict[str, Any]:
    """Run SpeciesNet prediction on a single image (local path or URL)."""
    instance = {"filepath": filepath}
    if country:
        instance["country"] = country
    if admin1_region:
        instance["admin1_region"] = admin1_region
    if latitude is not None:
        instance["latitude"] = latitude
    if longitude is not None:
        instance["longitude"] = longitude

    result = speciesnet_model.predict(
        instances_dict={"instances": [instance]},
        run_mode="single_thread",
        progress_bars=False,
    )

    if result and result.get("predictions"):
        return result["predictions"][0]
    return {"filepath": filepath, "failures": ["UNKNOWN"]}


def group_from_taxonomy(taxonomy_str: str) -> Optional[str]:
    """Look up the plain-English group name, falling back through family → order → class."""
    parts = taxonomy_str.split(';')
    family = parts[3] if len(parts) > 3 else ''
    order  = parts[2] if len(parts) > 2 else ''
    cls    = parts[1] if len(parts) > 1 else ''
    return FAMILY_GROUPS.get(family) or ORDER_GROUPS.get(order) or CLASS_GROUPS.get(cls)


def parse_label(taxonomy_str: str) -> str:
    """Extract common name from a SpeciesNet taxonomy string.

    Format: <uuid>;<class>;<order>;<family>;<genus>;<species>;<common_name>
    Returns the last non-empty segment, e.g. 'capybara', 'blank', 'animal'.
    """
    for part in reversed(taxonomy_str.split(';')):
        if part.strip():
            return part.strip().replace(' ', '_')
    return taxonomy_str


def format_predictions(
    prediction: Dict[str, Any],
    img_width: int,
    img_height: int,
) -> Optional[list]:
    """Reformat a raw SpeciesNet prediction into a flat list of prediction objects.

    Returns one entry per detection (matching YOLO's format), with bbox promoted
    to the top level of each prediction. Returns None if below CONFIDENCE_THRESHOLD,
    or a list with one bbox-less entry if there are no detections (e.g. blank frame).
    """
    if prediction.get("failures"):
        return [{"failures": prediction["failures"]}]

    # Bail out early if top-level confidence is below threshold
    if "prediction" not in prediction:
        return None
    confidence = round(prediction["prediction_score"], 4)
    if confidence < CONFIDENCE_THRESHOLD:
        return None

    taxonomy_str = prediction["prediction"]

    # Build the shared base fields (same for every detection)
    base = {
        "label": parse_label(taxonomy_str),
        "group": group_from_taxonomy(taxonomy_str) or parse_label(taxonomy_str),
        "confidence": confidence,
        "prediction_source": prediction.get("prediction_source"),
        "model_version": prediction.get("model_version"),
    }

    # Classifications: zip classes+scores, filter by threshold
    if "classifications" in prediction:
        classes = prediction["classifications"]["classes"]
        scores = prediction["classifications"]["scores"]
        base["classifications"] = [
            {"label": parse_label(cls), "score": round(score, 4)}
            for cls, score in zip(classes, scores)
            if score >= CONFIDENCE_THRESHOLD
        ]

    # Flatten: one prediction entry per detection, bbox at the top level
    detections = [
        det for det in prediction.get("detections", [])
        if det["conf"] >= CONFIDENCE_THRESHOLD
    ]
    if detections:
        return [
            {
                **base,
                "bbox": {
                    "x": round(det["bbox"][0] * img_width),
                    "y": round(det["bbox"][1] * img_height),
                    "width": round(det["bbox"][2] * img_width),
                    "height": round(det["bbox"][3] * img_height),
                },
            }
            for det in detections
        ]

    # No detections (blank frame, undetected animal, etc.) - return base without bbox
    return [base]


def create_response(
    prediction: Dict[str, Any],
    img_width: int,
    img_height: int,
    processing_time: float,
) -> Dict[str, Any]:
    """Create standardized API response."""
    has_failures = bool(prediction.get("failures"))
    formatted = format_predictions(prediction, img_width, img_height)
    return {
        "service": "speciesnet",
        "status": "error" if has_failures else "success",
        "predictions": formatted if formatted else [],
        "metadata": {
            "processing_time": round(processing_time, 3),
            "model_info": {
                "model": MODEL_NAME,
                "framework": "SpeciesNet (Google Camera Trap AI)",
            },
        },
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

app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

CORS(app, origins=["*"], methods=["GET", "POST", "OPTIONS"])
logger.info("SpeciesNet service: CORS enabled")


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
    """Health check endpoint."""
    model_status = "loaded" if speciesnet_model else "not_loaded"
    return jsonify({
        "status": "healthy",
        "model_status": model_status,
        "model": MODEL_NAME,
    })


@app.route('/analyze', methods=['GET', 'POST'])
@app.route('/v3/analyze', methods=['GET', 'POST'])
def analyze():
    """Unified analyze endpoint - accepts file upload, URL, or local file path.

    Optional geo parameters (query string or form fields):
        country       - ISO 3166-1 alpha-3 country code (e.g. 'USA', 'AUS')
        admin1_region - ISO 3166-2 region code (e.g. 'CA' for California)
        latitude      - float
        longitude     - float
    """
    start_time = time.time()

    def error_response(message: str, status_code: int = 400):
        return jsonify({
            "service": "speciesnet",
            "status": "error",
            "predictions": [],
            "error": {"message": message},
            "metadata": {"processing_time": round(time.time() - start_time, 3)},
        }), status_code

    try:
        # Collect geo parameters from query string or form data
        country = request.args.get('country') or request.form.get('country')
        admin1_region = request.args.get('admin1_region') or request.form.get('admin1_region')

        latitude = None
        longitude = None
        lat_str = request.args.get('latitude') or request.form.get('latitude')
        lon_str = request.args.get('longitude') or request.form.get('longitude')
        if lat_str:
            try:
                latitude = float(lat_str)
            except ValueError:
                return error_response("Invalid latitude value")
        if lon_str:
            try:
                longitude = float(lon_str)
            except ValueError:
                return error_response("Invalid longitude value")

        geo = dict(country=country, admin1_region=admin1_region,
                   latitude=latitude, longitude=longitude)

        if request.method == 'POST' and is_raw_image_request():
            file_data = request.get_data(cache=False)
            if not file_data:
                return error_response("No image body provided")
            if len(file_data) > MAX_FILE_SIZE:
                return error_response("File too large")

            img_width, img_height = image_size_from_bytes(file_data)
            ext = ext_from_content_type(request.content_type or '')
            with temp_image_file(file_data, ext) as filepath:
                prediction = run_prediction(filepath, **geo)

        # --- File upload (multipart POST) ---
        elif request.method == 'POST' and 'file' in request.files:
            uploaded_file = request.files['file']
            if uploaded_file.filename == '':
                return error_response("No file selected")
            if not is_allowed_file(uploaded_file.filename):
                return error_response("File type not allowed")

            file_data = uploaded_file.read()
            if len(file_data) > MAX_FILE_SIZE:
                return error_response("File too large")

            img_width, img_height = image_size_from_bytes(file_data)
            ext = '.' + uploaded_file.filename.rsplit('.', 1)[1].lower()
            with temp_image_file(file_data, ext) as filepath:
                prediction = run_prediction(filepath, **geo)

        # --- URL (we download it so we have the bytes for dimensions) ---
        elif request.args.get('url'):
            url = request.args.get('url')
            if not (url.startswith('http://') or url.startswith('https://')):
                return error_response("URL must start with http:// or https://")

            try:
                resp = requests.get(
                    url,
                    headers={'User-Agent': 'SpeciesNet API (github.com/google/cameratrapai)'},
                    timeout=30,
                )
                resp.raise_for_status()
            except requests.exceptions.RequestException as e:
                return error_response(f"Failed to download image: {e}")

            file_data = resp.content
            if len(file_data) > MAX_FILE_SIZE:
                return error_response(f"Image too large. Max size: {MAX_FILE_SIZE // (1024 * 1024)}MB")

            img_width, img_height = image_size_from_bytes(file_data)
            ext = ext_from_content_type(resp.headers.get('content-type', ''), url)
            with temp_image_file(file_data, ext) as filepath:
                prediction = run_prediction(filepath, **geo)

        # --- Local file path ---
        elif request.args.get('file'):
            file = request.args.get('file')
            if not os.path.exists(file):
                return error_response(f"File not found: {file}")
            if not is_allowed_file(file):
                return error_response("File type not allowed")

            img_width, img_height = image_size_from_path(file)
            prediction = run_prediction(file, **geo)

        else:
            return error_response(
                "Must provide a 'url' or 'file' query parameter, or POST a file"
            )

        return jsonify(create_response(prediction, img_width, img_height, time.time() - start_time))

    except Exception as e:
        logger.error(f"Analyze error: {e}", exc_info=True)
        return error_response(f"Internal error: {str(e)}", 500)


if __name__ == '__main__':
    logger.info("Starting SpeciesNet service...")

    load_family_groups()
    model_loaded = initialize_speciesnet()

    if not model_loaded:
        logger.error("Failed to initialize SpeciesNet model. Service cannot function.")
        exit(1)

    host = "127.0.0.1" if PRIVATE else "0.0.0.0"

    logger.info(f"Starting SpeciesNet service on {host}:{PORT}")
    logger.info(f"Private mode: {PRIVATE}")
    logger.info(f"Model: {MODEL_NAME}")

    app.run(
        host=host,
        port=PORT,
        debug=False,
        use_reloader=False,
        threaded=True,
    )
