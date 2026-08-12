# Detectron2 Object Detection Service

**Port**: 7771  
**Framework**: Facebook AI Research Detectron2 (Faster R-CNN)  
**Purpose**: Advanced object detection and instance segmentation with bounding boxes  
**Status**: ✅ Active

## Overview

Detectron2 provides state-of-the-art object detection using Facebook AI Research's Detectron2 framework. The service analyzes images and detects objects from 80 COCO classes, returning precise bounding boxes, confidence scores, and emoji mappings for each detection.

## Features

- **Modern V3 API**: Clean, unified endpoint with intuitive parameters
- **Unified Input Handling**: Single endpoint for both URL and file path analysis
- **Advanced Detection**: 80 COCO classes with precise bounding box coordinates
- **IoU Filtering**: Intelligent overlapping detection removal for cleaner results
- **Emoji Integration**: Automatic word-to-emoji mapping using local dictionary
- **GPU Acceleration**: CUDA-optimized inference with FP16 precision support
- **Security**: File validation, size limits, secure cleanup
- **Thread Safety**: Model inference locking for concurrent requests

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- 12GB+ RAM (16GB+ recommended for optimal performance)
- GPU with 8GB+ VRAM recommended

### 1. Environment Setup

```bash
# Navigate to Detectron2 directory
cd /home/sd/animal-farm/detectron2

# Create virtual environment
python3 -m venv detectron2_venv

# Activate virtual environment
source detectron2_venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Detectron2 Installation

Install Detectron2 with CUDA support:

```bash
# For CUDA 11.8 (adjust version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Detectron2
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Verify installation
python -c "import detectron2; print(detectron2.__version__)"
```

### 3. Model Configuration

The service uses Faster R-CNN with ResNet-50 FPN backbone:

```bash
# Config file location (automatically found)
/home/sd/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml

# Model weights (auto-downloaded on first run)
# detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl
```

## Configuration

### Environment Variables (.env)

Create a `.env` file in the detectron2 directory:

```bash
# Service Configuration
PORT=7771                    # Service port (default: 7771)
PRIVATE=False               # Access mode (False=public, True=localhost-only)

# Configuration Updates (GitHub-first pattern)
AUTO_UPDATE=true           # Refresh emoji mappings from GitHub on startup
TIMEOUT=10.0               # Timeout for remote config downloads (seconds)
```

### Configuration Details

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PORT` | Yes | - | Service listening port |
| `PRIVATE` | Yes | - | Access control (False=public, True=localhost-only) |
| `AUTO_UPDATE` | No | `true` | Refresh emoji mappings from GitHub on startup, then cache locally |
| `TIMEOUT` | No | `10.0` | Timeout for remote config downloads |

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
  "config_file": "/home/sd/detectron2/configs/...",
  "confidence_threshold": 0.5,
  "coco_classes_loaded": 80,
  "framework": "Detectron2"
}
```

### Unified Analysis (Recommended)
```bash
GET /analyze?url=<image_url>
GET /analyze?file=<file_path>
POST /analyze (with file upload)
```

**Parameters:**
- `url` (string): Image URL to analyze
- `file` (string): Local file path to analyze

**Note:** Exactly one parameter (`url` or `file`) must be provided.

**Response:**
```json
{
  "service": "detectron2",
  "status": "success",
  "predictions": [
    {
      "label": "person",
      "confidence": 0.998,
      "bbox": {
        "x": 256,
        "y": 9,
        "width": 384,
        "height": 440
      },
      "emoji": "🧑"
    },
    {
      "label": "person",
      "confidence": 0.996,
      "bbox": {
        "x": 6,
        "y": 23,
        "width": 232,
        "height": 429
      },
      "emoji": "🧑"
    },
    {
      "label": "teddy bear",
      "confidence": 0.991,
      "bbox": {
        "x": 260,
        "y": 162,
        "width": 115,
        "height": 113
      },
      "emoji": "🧸"
    }
  ],
  "metadata": {
    "processing_time": 0.153,
    "model_info": {
      "framework": "Facebook AI Research"
    }
  }
}
```

### V2 Compatibility Routes
```bash
GET /v2/analyze?image_url=<url>      # Translates to V3 url parameter
GET /v2/analyze_file?file_path=<path> # Translates to V3 file parameter
```

These endpoints maintain backward compatibility by translating parameters and calling the V3 endpoint internally.

## Service Management

### Manual Startup
```bash
# Activate virtual environment
source detectron2_venv/bin/activate

# Start service
python REST.py
```

### Systemd Service
```bash
# Install service file
sudo cp services/detectron-api.service /etc/systemd/system/

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable detectron-api.service
sudo systemctl start detectron-api.service

# Check status
sudo systemctl status detectron-api.service
```

## Performance Optimization

### Hardware Requirements

**Minimum:**
- 8GB RAM
- GTX 1060 6GB or equivalent
- 4-core CPU

**Recommended:**
- 16GB+ RAM
- RTX 3070 8GB or better
- 8-core CPU
- SSD storage

### Performance Settings

The service includes several optimizations:
- **FP16 Precision**: 50% VRAM reduction with autocast
- **Batch Size Optimization**: ROI heads batch size reduced to 128
- **Detection Limits**: Maximum 20 detections per image
- **Image Resizing**: 512px max dimension for faster inference
- **IoU Filtering**: Threshold 0.3 for overlapping detection removal

## Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `Model not loaded` | Detectron2 initialization failed | Check CUDA installation and model files |
| `Config file not found` | Missing configuration file | Verify Detectron2 installation |
| `File too large` | Image exceeds MAX_FILE_SIZE | Raise MAX_FILE_SIZE or use a smaller file |
| `Invalid URL format` | Malformed image URL | Check URL syntax and accessibility |
| `CUDA not available` | GPU/CUDA setup issue | Install CUDA drivers or use CPU mode |

### Error Response Format
```json
{
  "service": "detectron2",
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

# URL analysis
response = requests.get('http://localhost:7771/analyze', 
                       params={'url': 'https://example.com/image.jpg'})
data = response.json()

# File analysis
response = requests.get('http://localhost:7771/analyze',
                       params={'file': '/path/to/image.jpg'})
data = response.json()

# POST file upload
with open('/path/to/image.jpg', 'rb') as f:
    response = requests.post('http://localhost:7771/analyze',
                           files={'file': f})
    data = response.json()

# Process detections
for prediction in data['predictions']:
    print(f"Detected {prediction['label']} with {prediction['confidence']:.3f} confidence")
    bbox = prediction['bbox']
    print(f"Location: ({bbox['x']}, {bbox['y']}) {bbox['width']}x{bbox['height']}")
```

### JavaScript
```javascript
// URL analysis
const response = await fetch('http://localhost:7771/analyze?url=' + 
                            encodeURIComponent('https://example.com/image.jpg'));

// POST file upload
const formData = new FormData();
formData.append('file', fileInput.files[0]);
const response = await fetch('http://localhost:7771/analyze', {
    method: 'POST',
    body: formData
});
const data = await response.json();

// Process detections
data.predictions.forEach(prediction => {
    console.log(`Detected ${prediction.label} (${prediction.confidence})`);
    const bbox = prediction.bbox;
    console.log(`Bounding box: ${bbox.x},${bbox.y} ${bbox.width}x${bbox.height}`);
});
```

## Troubleshooting

### Installation Issues
- **ImportError**: Ensure Detectron2 is properly installed for your CUDA version
- **Config errors**: Verify Detectron2 repository structure and config files
- **CUDA errors**: Check PyTorch and Detectron2 CUDA compatibility

### Runtime Issues
- **Model loading fails**: Check GPU memory availability and CUDA installation
- **Slow inference**: Enable FP16 precision or reduce image resolution
- **Memory errors**: Reduce batch size or use CPU mode

### Performance Issues
- **High memory usage**: Enable FP16 precision with `USE_HALF_PRECISION = True`
- **Slow responses**: Check GPU utilization and consider multi-GPU setup
- **Detection quality**: Adjust `CONFIDENCE_THRESHOLD` (default: 0.5)

## Security Considerations

### Access Control
- Set `PRIVATE=True` for localhost-only access
- Use reverse proxy (nginx) for production deployment
- Implement rate limiting for public endpoints

### File Security
- File size limits enforced (32MB default)
- Allowed extensions validated
- Temporary files automatically cleaned up
- Path traversal protection enabled

## Detectable Objects (80 COCO Classes)

The service detects these object categories:

**People & Animals:**
person, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe

**Vehicles:**
bicycle, car, motorcycle, airplane, bus, train, truck, boat

**Outdoor Objects:**
traffic light, fire hydrant, stop sign, parking meter, bench

**Sports & Recreation:**
frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket

**Kitchen & Dining:**
bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake

**Furniture & Household:**
chair, couch, potted plant, bed, dining table, toilet

**Electronics:**
tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator

**Personal Items:**
backpack, umbrella, handbag, tie, suitcase, book, clock, vase, scissors, teddy bear, hair drier, toothbrush

---

*Generated with Animal Farm ML Platform v3.0 - Facebook AI Research Detectron2 Integration*
