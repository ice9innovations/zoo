# RTMDet Object Detection Service

**Port**: 7792
**Framework**: MMDetection RTMDet (Real-Time Multi-Object Detection Transformer)
**Purpose**: High-performance real-time object detection with emoji mapping
**Status**: ✅ Active

## Overview

RTMDet provides state-of-the-art real-time object detection using the MMDetection RTMDet framework. The service analyzes images to detect and localize objects with bounding boxes, automatically mapping detected classes to relevant emojis for enhanced user experience.

## Features

- **Modern V3 API**: Clean, unified endpoint with intuitive parameters
- **Unified Input Handling**: Single endpoint for URL, file path, and file upload analysis
- **Real-Time Detection**: High-performance object detection optimized for speed
- **Emoji Integration**: Automatic class-to-emoji mapping using local dictionary
- **Multiple Models**: Support for RTMDet variants (tiny, small, medium, large)
- **COCO Classes**: 80 object classes from the COCO dataset
- **Security**: File validation, size limits, secure cleanup
- **Performance**: GPU acceleration with CUDA support
- **Shiny Detection**: Fun 1/2500 chance for special "shiny" detections

## Installation

### Prerequisites

- Python 3.11+
- CUDA 12.1+ (for GPU acceleration)
- 8GB+ RAM (16GB+ recommended)
- 4GB+ storage (for models and dependencies)

### Method 1: Docker (Recommended)

```bash
# Navigate to RTMDet directory
cd /home/sd/animal-farm/rtmdet

# Build and start with Docker Compose
docker-compose up --build

# Or run in background
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop service
docker-compose down
```

### Method 2: Manual Installation

#### 1. Environment Setup

```bash
# Navigate to RTMDet directory
cd /home/sd/animal-farm/rtmdet

# Create virtual environment
python3 -m venv rtmdet_venv

# Activate virtual environment
source rtmdet_venv/bin/activate
```

#### 2. Dependency Installation (Critical Order)

**⚠️ Important**: Install dependencies in this exact order to avoid conflicts:

```bash
# Clean any existing installations
uv pip uninstall torch torchvision numpy openmim mmengine mmcv mmdet

# Install PyTorch with CUDA support
uv pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 --index-url https://download.pytorch.org/whl/cu121
uv pip install numpy==1.26.4

# Install OpenMMLab installer
uv pip install openmim

# Install MMDetection ecosystem (in order)
mim install mmengine
mim install "mmcv==2.1.0"
mim install mmdet

# Install additional dependencies
uv pip install flask flask-cors python-dotenv Pillow requests opencv-python==4.11.0
```

## Configuration

### Environment Variables (.env)

Create a `.env` file in the RTMDet directory:

```bash
# Service Configuration
PORT=7792                    # Service port (default: 7792)
PRIVATE=false               # Access mode (false=public, true=localhost-only)

# Model Configuration
CONFIDENCE_THRESHOLD=0.25   # Minimum confidence for detections (0.0-1.0)

# Emoji Configuration
AUTO_UPDATE=true            # Auto-download emoji mappings from GitHub
TIMEOUT=10.0               # Timeout for GitHub emoji downloads
```

### Configuration Details

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PORT` | Yes | - | Service listening port |
| `PRIVATE` | Yes | - | Access control (false=public, true=localhost-only) |
| `CONFIDENCE_THRESHOLD` | No | 0.25 | Minimum confidence score for detections |
| `AUTO_UPDATE` | No | true | Enable automatic emoji mapping updates |
| `TIMEOUT` | No | 10.0 | Timeout for external requests |

### Model Selection

RTMDet automatically downloads and selects the best available model variant:

| Model | Size | Accuracy | Speed | Use Case |
|-------|------|----------|-------|----------|
| `rtmdet_tiny` | ~5MB | Good | Fastest | Mobile/edge devices |
| `rtmdet_s` | ~20MB | Better | Fast | Balanced performance |
| `rtmdet_m` | ~50MB | High | Moderate | Production quality |
| `rtmdet_l` | ~100MB | Highest | Slower | Maximum accuracy |

## API Endpoints

### Health Check

```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "rtmdet",
  "capabilities": ["object_detection", "bbox_extraction"],
  "models": {
    "object_detection": {
      "status": "ready",
      "framework": "MMDetection RTMDet",
      "confidence_threshold": 0.25,
      "classes_loaded": 80
    }
  },
  "supported_classes": 80,
  "timestamp": 1705123456.789
}
```

### Get Supported Classes

```bash
GET /classes
```

**Response:**
```json
{
  "classes": ["person", "bicycle", "car", "motorcycle", "airplane", "..."],
  "total_classes": 80,
  "framework": "MMDetection RTMDet"
}
```

### Analyze Image (Unified Endpoint)

The unified `/analyze` endpoint accepts URL, file path, or file upload:

#### Analyze Image from URL
```bash
GET /analyze?url=<image_url>
```

**Example:**
```bash
curl "http://localhost:7792/analyze?url=https://example.com/image.jpg"
```

#### Analyze Image from File Path
```bash
GET /analyze?file=<file_path>
```

**Example:**
```bash
curl "http://localhost:7792/analyze?file=/path/to/image.jpg"
```

#### POST Request (File Upload)
```bash
POST /analyze
Content-Type: multipart/form-data
```

**Example:**
```bash
curl -X POST -F "file=@/path/to/image.jpg" http://localhost:7792/analyze
```

**Input Validation:**
- Exactly one parameter must be provided (`url` OR `file`)
- Cannot provide both parameters simultaneously
- Returns error if neither parameter is provided

**Response Format:**
```json
{
  "service": "rtmdet",
  "status": "success",
  "predictions": [
    {
      "label": "airplane",
      "confidence": 0.928,
      "emoji": "✈️",
      "bbox": {
        "x": 42,
        "y": 97,
        "width": 572,
        "height": 220
      }
    },
    {
      "label": "person",
      "confidence": 0.296,
      "emoji": "🧑",
      "bbox": {
        "x": 479,
        "y": 174,
        "width": 17,
        "height": 13
      },
      "shiny": true
    }
  ],
  "metadata": {
    "processing_time": 0.518,
    "model_info": {
      "framework": "MMDetection RTMDet"
    }
  }
}
```

## Service Management

### Docker Management

```bash
# Start service
docker-compose up -d

# View logs
docker-compose logs -f rtmdet

# Restart service
docker-compose restart rtmdet

# Stop service
docker-compose down

# Update and rebuild
docker-compose down && docker-compose up --build -d
```

### Manual Startup

```bash
# Start service
cd /home/sd/animal-farm/rtmdet
source rtmdet_venv/bin/activate
python REST.py
```

## Supported Formats

### Input Formats
- **Images**: PNG, JPG, JPEG, GIF, BMP, WebP
- **Max Size**: 32MB default
- **Input Methods**: URL, file upload, local path

### Object Classes (COCO Dataset)
**People & Animals**: person, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
**Vehicles**: bicycle, car, motorcycle, airplane, bus, train, truck, boat
**Objects**: chair, couch, bed, dining table, tv, laptop, mouse, remote, keyboard, cell phone
**Sports**: sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket
**Food**: bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake
**And more**: 80 total classes from COCO dataset

### Output Features
- **Object Detection**: Bounding box coordinates and dimensions
- **Confidence Scores**: Detection confidence (0.0-1.0)
- **Class Labels**: Human-readable object names
- **Emoji Mapping**: Automatic emoji assignment for detected objects
- **Shiny Detection**: Rare special detections (1/2500 chance)

## Performance Optimization

### Hardware Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|--------|
| GPU | GTX 1060 6GB | RTX 3080+ | CUDA 12.1+ required |
| RAM | 8GB | 16GB+ | Depends on model size |
| Storage | 4GB | 10GB+ | Models + dependencies |

### Optimization Settings

```python
# Model selection (in rtmdet_analyzer.py)
confidence_threshold = 0.25  # Lower = more detections
device = 'cuda'             # Use 'cpu' for CPU-only

# Performance tuning
max_detections = 100        # Limit output size
iou_threshold = 0.45       # Non-maximum suppression
```

## Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| "RTMDet analyzer not initialized" | Model loading failed | Check dependencies, restart service |
| "File too large" | File exceeds MAX_FILE_SIZE | Raise MAX_FILE_SIZE or use a smaller file |
| "Invalid URL" | Malformed URL | Check URL format |
| "MMDetection dependencies not available" | Installation issue | Follow exact dependency order |
| "CUDA out of memory" | Insufficient VRAM | Use smaller model or CPU mode |

### Error Response Format

```json
{
  "service": "rtmdet",
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
    "http://localhost:7792/analyze",
    params={"url": "https://example.com/image.jpg"}
)

# File input
response = requests.get(
    "http://localhost:7792/analyze",
    params={"file": "/path/to/image.jpg"}
)

# POST file upload
with open('/path/to/image.jpg', 'rb') as f:
    response = requests.post(
        "http://localhost:7792/analyze",
        files={'file': f}
    )

result = response.json()
if result["status"] == "success":
    for prediction in result["predictions"]:
        label = prediction["label"]
        confidence = prediction["confidence"]
        emoji = prediction.get("emoji", "")
        bbox = prediction["bbox"]
        shiny = prediction.get("shiny", False)

        print(f"{emoji} {label}: {confidence:.3f}")
        if shiny:
            print("✨ SHINY DETECTION! ✨")
        print(f"  Location: ({bbox['x']}, {bbox['y']}) - {bbox['width']}x{bbox['height']}")
```

### JavaScript Integration

```javascript
// URL input
const response = await fetch(
  'http://localhost:7792/analyze?url=https://example.com/image.jpg'
);

// File input
const response = await fetch(
  'http://localhost:7792/analyze?file=/path/to/image.jpg'
);

// POST file upload
const formData = new FormData();
formData.append('file', fileInput.files[0]);
const response = await fetch('http://localhost:7792/analyze', {
  method: 'POST',
  body: formData
});

const result = await response.json();
if (result.status === 'success') {
  result.predictions.forEach(prediction => {
    const { label, confidence, emoji, bbox, shiny } = prediction;
    console.log(`${emoji} ${label}: ${confidence.toFixed(3)}`);
    if shiny) console.log('✨ SHINY DETECTION! ✨');
    console.log(`Location: (${bbox.x}, ${bbox.y}) - ${bbox.width}x${bbox.height}`);
  });
}
```

## Troubleshooting

### Installation Issues

**Problem**: Dependency conflicts with PyTorch/MMDetection
```bash
# Solution: Follow exact installation order
uv pip uninstall torch torchvision numpy openmim mmengine mmcv mmdet
# Then reinstall following the exact sequence in installation section
```

**Problem**: "checkpoint is None" warning
```bash
# This indicates the model is using random weights instead of trained weights
# Solution: Service now automatically downloads proper checkpoints
# If issue persists, clear any cached model files and restart
```

**Problem**: CUDA not available
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.version.cuda)"

# Check CUDA version
nvidia-smi
```

### Runtime Issues

**Problem**: Service fails to start
```bash
# Check port availability
netstat -tlnp | grep 7792

# Check environment variables
grep -v '^#' .env

# Check logs for detailed error messages
docker-compose logs rtmdet  # For Docker
# Or check console output for manual start
```

**Problem**: Poor detection results
```bash
# Check model loading
curl http://localhost:7792/health

# Verify confidence threshold
# Lower values = more detections, higher false positives
# Higher values = fewer detections, higher precision
```

**Problem**: Emoji mapping failures
```bash
# Check emoji file exists
ls -la emoji_mappings.json

# Check cached mapping file
ls -la emoji_mappings.json

# Verify AUTO_UPDATE setting in .env
```

### Performance Issues

**Problem**: Slow inference
- Use smaller model (rtmdet_tiny or rtmdet_s)
- Reduce image resolution before processing
- Increase confidence threshold to reduce post-processing
- Use CPU mode for small workloads

**Problem**: Memory errors
- Use smaller RTMDet model variant
- Reduce image size before processing
- Monitor GPU memory usage with `nvidia-smi`
- Ensure sufficient system RAM

**Problem**: No detections found
- Lower confidence threshold (e.g., 0.1 instead of 0.25)
- Check if objects are in COCO dataset classes
- Verify image quality and resolution
- Test with known good images

## Security Considerations

### Access Control
- Set `PRIVATE=true` for localhost-only access
- Use reverse proxy with authentication for public access
- Validate all input URLs and file paths

### File Security
- Automatic cleanup of temporary files
- File type validation prevents executable uploads
- Size limits prevent DoS attacks
- Path traversal protection

### Model Security
- Models downloaded from official MMDetection sources
- Automatic checksum validation
- Secure HTTPS downloads

---

**Documentation Version**: 1.0
**Last Updated**: 2025-09-18
**Service Version**: Production
**Maintainer**: Animal Farm ML Team
