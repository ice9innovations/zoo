# RT-DETRv2 Object Detection Service

**Port**: 7781
**Framework**: RT-DETRv2 (Real-Time Detection Transformer v2)
**Purpose**: Advanced real-time object detection with transformer architecture and emoji mapping
**Status**: ✅ Active

## Overview

RT-DETRv2 provides state-of-the-art real-time object detection using the RT-DETRv2 (Real-Time Detection Transformer v2) model. This service detects 80 COCO object classes in images with confidence scores, bounding boxes, proper NMS filtering, and automatic emoji mapping for enhanced user experience.

## Features

- **Modern V3 API**: Clean, unified endpoint with intuitive parameters
- **Unified Input Handling**: Single endpoint for URL, file path, and upload analysis
- **COCO Object Detection**: Detects 80 common object classes
- **Proper NMS Filtering**: Built-in Non-Maximum Suppression for clean, non-overlapping detections
- **Emoji Integration**: Automatic word-to-emoji mapping using local dictionary
- **GPU Acceleration**: CUDA support for fast inference with fallback to CPU
- **Confidence Filtering**: Configurable detection thresholds
- **In-Memory Processing**: No temporary file storage, all processing in RAM
- **Multiple Model Sizes**: Support for rtdetrv2_s, rtdetrv2_m, rtdetrv2_l, rtdetrv2_x variants
- **Torch Hub Integration**: Clean model loading with automatic weight downloads

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0.1+
- CUDA-compatible GPU (recommended) or CPU
- 8GB+ RAM (16GB+ recommended for GPU)
- 10GB+ disk space for models
- Git (for cloning RT-DETRv2 repository)

### 1. Environment Setup

```bash
# Navigate to RT-DETRv2 directory
cd /home/sd/animal-farm/rtdetr2

# Create virtual environment
python3 -m venv rtdetrv2_venv

# Activate virtual environment
source rtdetrv2_venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. RT-DETRv2 Repository Installation

```bash
# Clone RT-DETRv2 repository
git clone https://github.com/supervisely-ecosystem/RT-DETRv2.git

# Verify installation
python test_setup.py
```

### 3. Hardware Configuration

```bash
# Verify CUDA availability (optional)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# Check GPU memory
nvidia-smi  # If CUDA GPU available
```

## Configuration

### Environment Variables (.env)

Create a `.env` file in the rtdetr2 directory:

```bash
# Service Configuration
PORT=7781                           # Service port
PRIVATE=False                       # Access mode (False=public, True=localhost-only)

# Configuration Updates (GitHub-first pattern)
AUTO_UPDATE=true                    # Auto-update emoji mappings from GitHub
TIMEOUT=2.0                         # Timeout for requests

# Detection Settings
CONFIDENCE_THRESHOLD=0.25           # Minimum confidence for detections

# Model Configuration
MODEL_SIZE=rtdetrv2_s              # Model variant (rtdetrv2_s, rtdetrv2_m_r34, rtdetrv2_m_r50, rtdetrv2_l, rtdetrv2_x)
```

### Configuration Details

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PORT` | Yes | - | Service listening port |
| `PRIVATE` | Yes | - | Access control (False=public, True=localhost-only) |
| `AUTO_UPDATE` | No | true | Auto-update emoji mappings from GitHub |
| `CONFIDENCE_THRESHOLD` | No | 0.25 | Minimum detection confidence |
| `MODEL_SIZE` | No | rtdetrv2_s | Model variant (s=fastest, x=most accurate) |

### Model Size Options

| Model | Speed | Accuracy | GPU Memory | Use Case |
|-------|-------|----------|------------|----------|
| `rtdetrv2_s` | Fastest | Good | ~2GB | Development/Fast inference |
| `rtdetrv2_m_r34` | Fast | Better | ~3GB | Balanced performance |
| `rtdetrv2_m_r50` | Medium | Better | ~4GB | Balanced accuracy |
| `rtdetrv2_l` | Slower | High | ~6GB | Production quality |
| `rtdetrv2_x` | Slowest | Highest | ~8GB | Maximum accuracy |

## API Endpoints

### Health Check
```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "RT-DETRv2 Object Detection",
  "model": {
    "name": "RT-DETRv2 (rtdetrv2_s)",
    "status": "loaded",
    "device": "cuda"
  },
  "confidence_threshold": 0.25,
  "emoji_service": "local_file"
}
```

### V3 Unified Analysis (Recommended)
```bash
GET /analyze?url=<image_url>
GET /analyze?file=<file_path>
POST /analyze  # With file upload
```

**Parameters:**
- `url` (string): Image URL to analyze
- `file` (string): Local file path to analyze
- File upload via POST with multipart/form-data

**Note:** Exactly one input method must be provided.

**Response:**
```json
{
  "service": "rtdetrv2",
  "status": "success",
  "predictions": [
    {
      "label": "person",
      "confidence": 0.965,
      "bbox": {
        "x": 136,
        "y": 32,
        "width": 334,
        "height": 309
      },
      "emoji": "🧑"
    },
    {
      "label": "skateboard",
      "confidence": 0.9,
      "bbox": {
        "x": 167,
        "y": 215,
        "width": 82,
        "height": 106
      },
      "emoji": "🛹"
    }
  ],
  "metadata": {
    "processing_time": 0.052,
    "model_info": {
      "framework": "RT-DETRv2 PyTorch",
      "model_size": "rtdetrv2_s"
    }
  }
}
```

### V2 Compatibility Routes
```bash
GET /v2/analyze?image_url=<url>       # Translates to V3 url parameter
GET /v2/analyze_file?file_path=<path>  # Translates to V3 file parameter
```

## Service Management

### Manual Startup
```bash
# Activate virtual environment
source rtdetrv2_venv/bin/activate

# Start service
python REST.py
```

### Docker Deployment
```bash
# Build Docker image
docker build -t rtdetrv2-api .

# Run container
docker run -d \
  --name rtdetrv2-api \
  --gpus all \
  -p 7781:7781 \
  -e MODEL_SIZE=rtdetrv2_s \
  rtdetrv2-api
```

## Performance Optimization

### Hardware Requirements

**Minimum:**
- 8GB RAM
- 4-core CPU
- 10GB disk space

**Recommended:**
- 16GB+ RAM
- 8-core CPU
- RTX 3080 or better GPU
- NVMe SSD storage

### Model Performance Comparison

| Model | Inference Time | mAP | GPU Memory | Use Case |
|-------|----------------|-----|------------|----------|
| rtdetrv2_s | ~50ms | 48.1 | 2GB | Real-time applications |
| rtdetrv2_l | ~80ms | 53.0 | 6GB | Production quality |
| rtdetrv2_x | ~120ms | 54.8 | 8GB | Maximum accuracy |

### Optimization Features

- **Proper NMS**: Built-in Non-Maximum Suppression eliminates overlapping detections
- **Confidence Filtering**: Configurable threshold removes low-confidence predictions
- **GPU Acceleration**: Automatically detected and enabled
- **Memory Management**: Periodic GPU cache cleanup prevents memory leaks
- **In-Memory Processing**: No disk I/O for image processing

## Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `Model not loaded` | RT-DETRv2 model failed to load | Check RT-DETRv2 repository and GPU drivers |
| `File not found` | Invalid file path | Verify file exists and path is correct |
| `File too large` | Image exceeds MAX_FILE_SIZE | Raise MAX_FILE_SIZE or use a smaller file |
| `Invalid URL format` | Malformed image URL | Check URL syntax and accessibility |
| `URL does not point to an image` | Non-image content type | Ensure URL serves image content |

### Error Response Format
```json
{
  "service": "rtdetrv2",
  "status": "error",
  "predictions": [],
  "error": {"message": "Detailed error description"},
  "metadata": {"processing_time": 0.001}
}
```

## Integration Examples

### Python
```python
import requests

# Object detection from URL
response = requests.get('http://localhost:7781/analyze',
                       params={'url': 'https://example.com/image.jpg'})
data = response.json()

# Object detection from file
response = requests.get('http://localhost:7781/analyze',
                       params={'file': '/path/to/image.jpg'})
result = response.json()

# File upload
with open('image.jpg', 'rb') as f:
    response = requests.post('http://localhost:7781/analyze',
                           files={'file': f})
result = response.json()

# Process results (note: labels use underscores)
for prediction in result['predictions']:
    label = prediction['label']  # e.g., "baseball_glove", "traffic_light"
    confidence = prediction['confidence']
    bbox = prediction['bbox']
    emoji = prediction.get('emoji', '')
    print(f"{emoji} {label}: {confidence:.2f} at ({bbox['x']}, {bbox['y']})")
```

### JavaScript
```javascript
// Object detection from URL
const response = await fetch('http://localhost:7781/analyze?' +
    new URLSearchParams({url: 'https://example.com/image.jpg'}));
const data = await response.json();

// File upload
const formData = new FormData();
formData.append('file', fileInput.files[0]);
const uploadResponse = await fetch('http://localhost:7781/analyze', {
    method: 'POST',
    body: formData
});
const uploadResult = await uploadResponse.json();

// Process detections
data.predictions.forEach(prediction => {
    console.log(`${prediction.emoji || ''} ${prediction.label}: ${prediction.confidence}`);
    console.log(`Location: (${prediction.bbox.x}, ${prediction.bbox.y})`);
    console.log(`Size: ${prediction.bbox.width}x${prediction.bbox.height}`);
});
```

## Troubleshooting

### Installation Issues
- **RT-DETRv2 not found**: Ensure `git clone https://github.com/supervisely-ecosystem/RT-DETRv2.git` completed
- **PyTorch version**: RT-DETRv2 requires PyTorch 2.0.1+
- **CUDA not detected**: Install NVIDIA drivers and CUDA toolkit
- **Dependencies missing**: Use requirements.txt in virtual environment

### Runtime Issues
- **Model loading fails**: Run `python test_setup.py` to diagnose
- **Slow inference**: Ensure GPU is detected, check GPU memory
- **Memory errors**: Use smaller model size or increase system RAM
- **Connection refused**: Verify service is running on port 7781

### Performance Issues
- **Multiple detections**: RT-DETRv2 may detect shadows/reflections as legitimate objects
- **Low detection accuracy**: Try larger model size (rtdetrv2_l or rtdetrv2_x)
- **Too many false positives**: Increase confidence threshold in .env
- **Missing objects**: Decrease confidence threshold or try different model size

## Architecture Notes

### NMS Implementation
RT-DETRv2 uses a custom postprocessor with proper Non-Maximum Suppression:
- **Class-agnostic NMS**: Removes overlapping boxes regardless of class
- **Configurable threshold**: 0.5 IoU threshold for overlap detection
- **Score-based ranking**: Keeps highest confidence detections

### Memory Management
- **No temporary files**: All image processing in RAM using PIL and BytesIO
- **GPU cache cleanup**: Periodic memory cleanup prevents OOM errors
- **Efficient tensor handling**: Explicit GPU tensor deletion after processing

## Security Considerations

### Access Control
- Set `PRIVATE=True` for localhost-only access
- Use reverse proxy (nginx) for production deployment
- Implement rate limiting for public endpoints

### File Security
- File uploads processed in memory, no disk storage
- Size limits prevent resource exhaustion
- Only allowed image formats accepted
- Input validation for all parameters

## Supported Formats

### Input Formats
- **Images**: PNG, JPG, JPEG, GIF, BMP, WebP
- **Max Size**: 32MB default
- **Input Methods**: URL, file upload, local path

### Output Features
- **Object Detection**: 80 COCO object classes with bounding boxes
- **Confidence Scores**: Detection confidence values (0.0-1.0)
- **Emoji Mapping**: Automatic object-to-emoji conversion
- **Underscore Labels**: Backend-friendly class names (e.g., "baseball_glove")
- **NMS Filtering**: Clean, non-overlapping detection results

---

**Documentation Version**: 2.0
**Last Updated**: 2025-09-20
**Service Version**: Production
**Maintainer**: Animal Farm ML Team