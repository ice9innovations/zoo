# CLIP Image Classification Service

**Port**: 7778  
**Framework**: OpenAI CLIP (Contrastive Language-Image Pre-training)  
**Purpose**: AI-powered image classification with emoji mapping  
**Status**: ✅ Active

## Overview

CLIP provides state-of-the-art image classification using OpenAI's CLIP model. The service analyzes images against a comprehensive set of labels and returns the most relevant classifications with confidence scores, automatically mapping labels to relevant emojis for enhanced user experience.

## Features

- **Modern V3 API**: Clean, unified endpoint with intuitive parameters
- **Unified Input Handling**: Single endpoint for both URL and file path analysis
- **Emoji Integration**: Automatic label-to-emoji mapping using local dictionary
- **Comprehensive Labels**: Classification across multiple categories (animals, objects, people, etc.)
- **Confidence Filtering**: Configurable thresholds for prediction quality
- **Security**: File validation, size limits, secure cleanup
- **Performance**: GPU acceleration with CUDA/MPS support and FP16 optimization

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.2+ (for GPU acceleration)
- 4GB+ VRAM (8GB+ recommended for ViT-L models)
- 8GB+ RAM (16GB+ recommended)

### 1. Environment Setup

```bash
# Navigate to CLIP directory
cd /home/sd/animal-farm/CLIP

# Create virtual environment
python3 -m venv clip_venv

# Activate virtual environment
source clip_venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Model Installation

Install OpenAI CLIP:

```bash
# Install CLIP from official repository
pip install clip-by-openai

# Or install from source for latest features
pip install git+https://github.com/openai/CLIP.git
```

### 3. Label Configuration

The service automatically loads classification labels from `.txt` files in the `labels/` directory:

```bash
# Labels are loaded from all .txt files in labels/ folder
ls labels/
# animals.txt  objects.txt  people.txt  transport.txt  etc.
```

## Configuration

### Environment Variables (.env)

Create a `.env` file in the CLIP directory:

```bash
# Service Configuration
PORT=7778                       # Service port
PRIVATE=False                   # Access mode (False=public, True=localhost-only)

# CLIP Model Configuration  
CLIP_MODEL=ViT-B/32            # Model size (ViT-B/32, ViT-L/14, ViT-L/14@336px)
CLIP_CONFIDENCE_THRESHOLD=0.01  # Minimum confidence for predictions
CLIP_MAX_PREDICTIONS=10         # Maximum number of predictions to return

# Configuration Updates (GitHub-first pattern)
AUTO_UPDATE=true                # Refresh emoji mappings from GitHub on startup
API_TIMEOUT=10.0                # Timeout for remote config downloads (seconds)
```

### Configuration Details

| Variable | Required | Description |
|----------|----------|-------------|
| `PORT` | Yes | Service listening port |
| `PRIVATE` | Yes | Access control (False=public, True=localhost-only) |
| `CLIP_MODEL` | Yes | CLIP model variant to use |
| `CLIP_CONFIDENCE_THRESHOLD` | Yes | Minimum confidence score for predictions |
| `CLIP_MAX_PREDICTIONS` | Yes | Maximum number of predictions to return |
| `AUTO_UPDATE` | No | Refresh emoji mappings from GitHub on startup, then cache locally |
| `API_TIMEOUT` | No | Timeout for remote config downloads |

### Model Options

| Model | Size | VRAM | Speed | Accuracy | Description |
|-------|------|------|--------|----------|-------------|
| `ViT-B/32` | 151MB | ~2GB | Fastest | Good | Recommended for development |
| `ViT-B/16` | 338MB | ~4GB | Fast | Better | Higher resolution processing |
| `ViT-L/14` | 428MB | ~6GB | Slower | Best | Production quality |
| `ViT-L/14@336px` | 428MB | ~8GB | Slowest | Highest | Maximum accuracy |

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
  "device": "cuda:0",
  "num_labels": 1250
}
```

### Analyze Image (Unified Endpoint)

The unified `/analyze` endpoint accepts either URL or file path input:

#### Analyze Image from URL
```bash
GET /analyze?url=<image_url>
```

**Example:**
```bash
curl "http://localhost:7778/analyze?url=https://example.com/image.jpg"
```

#### Analyze Image from File Path
```bash
GET /analyze?file=<file_path>
```

**Example:**
```bash
curl "http://localhost:7778/analyze?file=/path/to/image.jpg"
```

#### POST Request (File Upload)
```bash
POST /analyze
Content-Type: multipart/form-data
```

**Example:**
```bash
curl -X POST -F "file=@/path/to/image.jpg" http://localhost:7778/analyze
```

**Input Validation:**
- Exactly one parameter must be provided (`url` OR `file`)
- Cannot provide both parameters simultaneously
- Returns error if neither parameter is provided

**Response Format:**
```json
{
  "service": "clip",
  "status": "success",
  "predictions": [
    {
      "label": "man",
      "confidence": 0.401,
      "emoji": "🧑"
    },
    {
      "label": "uniform",
      "confidence": 0.211,
      "emoji": "👔"
    },
    {
      "label": "child",
      "confidence": 0.132,
      "emoji": "🧑"
    }
  ],
  "metadata": {
    "processing_time": 0.16,
    "model_info": {
      "framework": "OpenAI"
    }
  }
}
```

## Service Management

### Manual Startup

```bash
# Start service
cd /home/sd/animal-farm/CLIP
./CLIP.sh
```

### Systemd Service

```bash
# Install service
sudo cp services/CLIP-api.service /etc/systemd/system/
sudo systemctl daemon-reload

# Start/stop service
sudo systemctl start CLIP-api
sudo systemctl stop CLIP-api

# Enable auto-start
sudo systemctl enable CLIP-api

# Check status
sudo systemctl status CLIP-api

# View logs
sudo journalctl -u CLIP-api -f
```

## Supported Formats

### Input Formats
- **Images**: PNG, JPG, JPEG, GIF, BMP, WebP
- **Max Size**: 32MB default
- **Input Methods**: URL, file upload, local path

### Output Features
- **Multi-label Classification**: Returns multiple relevant labels per image
- **Confidence Scores**: Probability scores for each prediction
- **Emoji Mapping**: Automatic label-to-emoji conversion
- **Configurable Filtering**: Adjustable confidence thresholds
- **Performance Metadata**: Processing time and model information

## Performance Optimization

### Hardware Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|--------|
| GPU | GTX 1060 6GB | RTX 3080+ | CUDA 11.2+ required |
| RAM | 8GB | 16GB+ | Depends on model and label count |
| Storage | 1GB | 2GB+ | Models + labels + temp files |

### Optimization Settings

```python
# Model configuration (in REST.py)
CLIP_MODEL = 'ViT-B/32'  # Faster inference
CLIP_CONFIDENCE_THRESHOLD = 0.05  # Filter low-confidence predictions
CLIP_MAX_PREDICTIONS = 5  # Limit output size

# GPU optimization
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.half()  # FP16 for 50% VRAM reduction
```

## Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| "Model not loaded" | CLIP not installed | Install clip-by-openai package |
| "File too large" | File exceeds MAX_FILE_SIZE | Raise MAX_FILE_SIZE or use a smaller file |
| "Invalid URL" | Malformed URL | Check URL format |
| "No labels loaded" | Missing label files | Ensure .txt files exist in labels/ |
| "CUDA out of memory" | Insufficient VRAM | Use smaller model or CPU mode |

### Error Response Format

```json
{
  "service": "clip",
  "status": "error",
  "predictions": [],
  "error": {"message": "File not found: /path/to/image.jpg"},
  "metadata": {"processing_time": 0.001}
}
```

## Integration Examples

### Python Integration

```python
import requests

# URL input
response = requests.get(
    "http://localhost:7778/analyze",
    params={"url": "https://example.com/image.jpg"}
)

# File input
response = requests.get(
    "http://localhost:7778/analyze",
    params={"file": "/path/to/image.jpg"}
)

# POST file upload
with open('/path/to/image.jpg', 'rb') as f:
    response = requests.post(
        "http://localhost:7778/analyze",
        files={'file': f}
    )

result = response.json()
if result["status"] == "success":
    predictions = result["predictions"]
    for pred in predictions:
        label = pred["label"]
        confidence = pred["confidence"]
        emoji = pred.get("emoji", "")
        print(f"{label} ({confidence:.3f}) {emoji}")
```

### JavaScript Integration

```javascript
// URL input
const response = await fetch(
  'http://localhost:7778/analyze?url=https://example.com/image.jpg'
);

// File input
const response = await fetch(
  'http://localhost:7778/analyze?file=/path/to/image.jpg'
);

// POST file upload
const formData = new FormData();
formData.append('file', fileInput.files[0]);
const response = await fetch('http://localhost:7778/analyze', {
  method: 'POST',
  body: formData
});

const result = await response.json();
if (result.status === 'success') {
  result.predictions.forEach(pred => {
    console.log(`${pred.label} (${pred.confidence.toFixed(3)}) ${pred.emoji || ''}`);
  });
}
```

## Troubleshooting

### Installation Issues

**Problem**: Import errors for CLIP
```bash
# Solution: Install CLIP package
pip install clip-by-openai

# Verify installation
python -c "import clip; print('CLIP installed successfully')"
```

**Problem**: CUDA not available
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Check CUDA version
nvidia-smi
```

### Runtime Issues

**Problem**: Service fails to start
```bash
# Check port availability
netstat -tlnp | grep 7778

# Check environment variables
grep -v '^#' .env

# Check logs
tail -f /var/log/syslog | grep CLIP
```

**Problem**: No predictions returned
```bash
# Check confidence threshold (may be too high)
# Lower CLIP_CONFIDENCE_THRESHOLD in .env

# Verify label files exist
ls -la labels/*.txt
```

**Problem**: Emoji mapping failures
```bash
# Check cached mapping file
ls -la emoji_mappings.json

# Check config loading settings
grep -E 'AUTO_UPDATE|API_TIMEOUT' .env
```

### Performance Issues

**Problem**: Slow inference
- Use smaller model (`ViT-B/32` instead of `ViT-L/14`)
- Enable FP16 precision for supported GPUs
- Reduce `CLIP_MAX_PREDICTIONS` to limit output processing

**Problem**: Memory errors
- Reduce image size before processing
- Use CPU instead of GPU for small workloads
- Ensure sufficient system RAM and VRAM

## Security Considerations

### Access Control
- Set `PRIVATE=True` for localhost-only access
- Use reverse proxy with authentication for public access
- Validate all input URLs and file paths

### File Security
- Automatic cleanup of temporary files
- File type validation prevents executable uploads
- Size limits prevent DoS attacks
- Path traversal protection

---

**Documentation Version**: 1.0  
**Last Updated**: 2025-08-13  
**Service Version**: Production  
**Maintainer**: Animal Farm ML Team
