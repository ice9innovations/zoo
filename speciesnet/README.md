# SpeciesNet Image Classification Service

**Framework**: Google SpeciesNet (Camera Trap AI)
**Purpose**: Wildlife species identification from camera trap images
**Status**: Active

## Overview

SpeciesNet runs a full detector + classifier + ensemble pipeline to identify wildlife species in images. It returns a primary prediction (common name, confidence score), filtered classification candidates, and pixel-coordinate bounding box detections. Geofenced results are available when location data is provided.

## Features

- **Unified API**: Single endpoint for URL, local file path, or file upload
- **Geolocation support**: Optional country/region/GPS coordinates for geofenced predictions
- **Confidence filtering**: Configurable threshold filters low-confidence results
- **Privacy-safe**: Image data processed in RAM only (`/dev/shm`) — never written to disk
- **CORS enabled**: Direct browser access supported

## Installation

### Prerequisites

- Python 3.10+
- SpeciesNet installed via pip (`pip install speciesnet`)
- Flask and flask-cors installed in the same environment

### 1. Environment Setup

```bash
cd /home/sd/speciesnet

# Activate virtual environment
source speciesnet_venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Model Download

SpeciesNet downloads model weights automatically from Kaggle on first run. Weights are cached locally (~1.5GB). A Kaggle account and API token are required for the first download.

```bash
# Set up Kaggle credentials (one-time)
mkdir -p ~/.kaggle
# Place your kaggle.json token file at ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

## Configuration

### Environment Variables (.env)

```bash
PORT=7778
PRIVATE=False
CONFIDENCE_THRESHOLD=0.5
# MODEL=kaggle:google/speciesnet/pyTorch/v4.0.2a/1  # optional, this is the default
```

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PORT` | Yes | - | Service listening port |
| `PRIVATE` | Yes | - | Access control (`False`=public, `True`=localhost-only) |
| `CONFIDENCE_THRESHOLD` | No | `0.5` | Minimum confidence to include a result |
| `MODEL` | No | `kaggle:google/speciesnet/pyTorch/v4.0.2a/1` | Model version to load |

## API Endpoints

### Health Check

```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_status": "loaded",
  "model": "kaggle:google/speciesnet/pyTorch/v4.0.2a/1"
}
```

### Analyze Image (Unified Endpoint)

```bash
GET  /analyze?url=<image_url>
GET  /analyze?file=<local_file_path>
POST /analyze  (multipart file upload)
```

#### Optional geo parameters (query string or form fields)

| Parameter | Type | Description |
|-----------|------|-------------|
| `country` | string | ISO 3166-1 alpha-3 code (e.g. `USA`, `AUS`, `KEN`) |
| `admin1_region` | string | ISO 3166-2 region (e.g. `CA` for California) |
| `latitude` | float | GPS latitude |
| `longitude` | float | GPS longitude |

Providing location data enables SpeciesNet's geofencing, which can improve accuracy by filtering out species not found in the given region.

**Examples:**

```bash
# Analyze from local file
curl "http://localhost:7778/analyze?file=/path/to/image.jpg"

# Analyze from URL
curl "http://localhost:7778/analyze?url=https://example.com/camera_trap.jpg"

# With geolocation
curl "http://localhost:7778/analyze?file=/path/to/image.jpg&country=KEN&latitude=-1.2&longitude=36.8"

# Upload a file
curl -X POST -F "file=@/path/to/image.jpg" http://localhost:7778/analyze

# Upload with geo params
curl -X POST \
  -F "file=@/path/to/image.jpg" \
  -F "country=USA" \
  -F "admin1_region=CA" \
  http://localhost:7778/analyze
```

**Successful Response:**
```json
{
  "service": "speciesnet",
  "status": "success",
  "predictions": [
    {
      "label": "capybara",
      "confidence": 0.9662,
      "bbox": {"x": 44, "y": 29, "width": 198, "height": 126},
      "classifications": [
        {"label": "capybara", "score": 0.9662}
      ],
      "prediction_source": "classifier",
      "model_version": "4.0.2a"
    }
  ],
  "metadata": {
    "processing_time": 0.468,
    "model_info": {
      "model": "kaggle:google/speciesnet/pyTorch/v4.0.2a/1",
      "framework": "SpeciesNet (Google Camera Trap AI)"
    }
  }
}
```

There is one entry in `predictions` per detected animal. If multiple animals are detected in a frame, each gets its own entry with its own `bbox`, all sharing the same top-level `label` and `confidence` (SpeciesNet classifies at the image level, not per animal).

Blank frames or images where no animal is detected above the threshold will have one entry without a `bbox`:
```json
{"label": "blank", "confidence": 0.9409, "classifications": [...], ...}
```

When the top-level confidence is below `CONFIDENCE_THRESHOLD`, `predictions` is an empty list:
```json
{"service": "speciesnet", "status": "success", "predictions": [], ...}
```

**Error Response:**
```json
{
  "service": "speciesnet",
  "status": "error",
  "predictions": [],
  "error": {"message": "File not found: /path/to/missing.jpg"},
  "metadata": {"processing_time": 0.001}
}
```

### Response Fields

| Field | Description |
|-------|-------------|
| `label` | Common name of the identified species (e.g. `capybara`, `animal`, `blank`) |
| `confidence` | Top-level prediction confidence, rounded to 4 decimal places |
| `bbox` | Pixel-coordinate bounding box `{x, y, width, height}` — omitted for blank frames |
| `classifications` | All classifications at or above `CONFIDENCE_THRESHOLD`, each with `label` and `score` |
| `prediction_source` | How the result was derived (see below) |
| `model_version` | SpeciesNet model version that produced the result |

### Prediction Sources

| Value | Meaning |
|-------|---------|
| `classifier` | Classifier confidence was high enough for a direct result |
| `detector` | Detection confidence drove the result (classifier uncertain) |
| `classifier+rollup_to_*` | Result rolled up the taxonomy tree due to low species-level confidence |
| `geofence_*` | Geofencing filtered or adjusted the result |

### Bounding Box Format

Bounding boxes are in **pixel coordinates** for the input image:

```json
{"x": 44, "y": 29, "width": 198, "height": 126}
```

Where `x`, `y` is the top-left corner of the box.

## Service Management

### Manual Startup

```bash
cd /home/sd/speciesnet
source speciesnet_venv/bin/activate
python REST.py
```

### Systemd

```bash
sudo systemctl start speciesnet-api
sudo systemctl stop speciesnet-api
sudo systemctl status speciesnet-api
sudo journalctl -u speciesnet-api -f
```

## Supported Image Formats

- **Input**: PNG, JPG/JPEG, TIFF/TIF, WebP
- **Max size**: 20MB
- **Temp storage**: RAM only (`/dev/shm`) — image data never written to disk

## Integration Examples

### Python

```python
import requests

# From local file path
resp = requests.get(
    "http://localhost:7778/analyze",
    params={"file": "/data/images/trap001.jpg"}
)

# From URL with geolocation
resp = requests.get(
    "http://localhost:7778/analyze",
    params={"url": "https://example.com/trap.jpg", "country": "KEN"}
)

# File upload with geo params
with open("/data/images/trap001.jpg", "rb") as f:
    resp = requests.post(
        "http://localhost:7778/analyze",
        files={"file": f},
        data={"country": "USA", "admin1_region": "CA"}
    )

result = resp.json()
if result["status"] == "success" and result["predictions"]:
    pred = result["predictions"][0]
    print(f"Species: {pred['label']}")
    print(f"Score:   {pred['confidence']}")
    print(f"Source:  {pred['prediction_source']}")
```

### JavaScript

```javascript
// From local file path
const resp = await fetch('http://localhost:7778/analyze?file=/data/images/trap001.jpg');

// File upload
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('country', 'USA');
const resp = await fetch('http://localhost:7778/analyze', {
  method: 'POST',
  body: formData
});

const result = await resp.json();
if (result.status === 'success' && result.predictions.length > 0) {
  const pred = result.predictions[0];
  console.log('Identified as:', pred.label, 'with score', pred.confidence);
}
```

## Error Reference

| Error | Cause | Solution |
|-------|-------|----------|
| `File not found` | Bad local path | Check the file path |
| `File type not allowed` | Unsupported extension | Use JPG, PNG, TIFF, or WebP |
| `File too large` | File > 20MB | Resize or compress the image |
| `Failed to download image` | Bad URL or network error | Verify the URL is accessible |
| `URL must start with http://` | Non-HTTP URL provided | Use a full http/https URL |
| `Model not loaded` | Startup failure | Check logs for Kaggle auth errors |

---

**Documentation Version**: 1.1
**Last Updated**: 2026-02-18
**Upstream**: https://github.com/google/cameratrapai
