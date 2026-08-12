# YOLO Open Images v7 Detection Service

**Port**: 7791  
**Framework**: Ultralytics YOLOv11  
**Purpose**: Open Images v7 object detection with bounding boxes and emoji mapping  
**Status**: ✅ Active

## Overview

This service provides Open Images v7 object detection using an Ultralytics YOLO model, returning confidence scores, bounding boxes, and emoji mappings for detected classes.

## Features

- **Modern V3 API**: Clean, unified endpoint with intuitive parameters
- **Unified Input Handling**: Single endpoint for both URL and file path analysis
- **Open Images v7 Detection**: Detects classes from the Open Images v7 training set
- **Advanced IoU Filtering**: Removes overlapping detections for cleaner results
- **Emoji Integration**: Automatic word-to-emoji mapping using local dictionary
- **GPU Acceleration**: CUDA and MPS support for fast inference
- **Confidence Filtering**: Configurable detection thresholds
- **Security**: File validation, size limits, secure cleanup

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended) or CPU
- 4GB+ RAM (8GB+ recommended)
- 10GB+ disk space for models

### 1. Environment Setup

```bash
# Navigate to yolo_oi7 directory
cd /home/sd/animal-farm/yolo_oi7

# Create virtual environment
python3 -m venv yolo_venv

# Activate virtual environment
source yolo_venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Model Installation

```bash
# YOLOv8 models are downloaded automatically on first use
# Available models (in order of accuracy vs speed):
# - yolov8n.pt (Nano - fastest)
# - yolov8s.pt (Small)
# - yolov8m.pt (Medium)
# - yolov8l.pt (Large)  
# - yolov8x.pt (Extra Large - most accurate)

# Pre-download models (optional)
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### 3. Hardware Configuration

```bash
# Verify CUDA availability (optional)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check GPU memory
nvidia-smi  # If CUDA GPU available
```

## Configuration

### Environment Variables (.env)

Create a `.env` file in the `yolo_oi7` directory:

```bash
# Service Configuration
PORT=7791                           # Service port
PRIVATE=False                       # Access mode (False=public, True=localhost-only)

# Configuration Updates (GitHub-first pattern)
AUTO_UPDATE=true                   # Refresh emoji mappings from GitHub on startup
TIMEOUT=10.0                       # Timeout for remote config downloads (seconds)
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
  "confidence_threshold": 0.25,
  "iou_threshold": 0.3,
  "supported_classes": 80
}
```

### Supported Classes
```bash
GET /classes
```

**Response:**
```json
{
  "classes": ["person", "bicycle", "car", "motorcycle", ...],
  "total_classes": 80
}
```

### V3 Unified Analysis (Recommended)
```bash
GET /analyze?url=<image_url>
GET /analyze?file=<file_path>
```

**Parameters:**
- `url` (string): Image URL to analyze
- `file` (string): Local file path to analyze

**Note:** Exactly one parameter (`url` or `file`) must be provided.

**Response:**
```json
{
  "service": "yolo",
  "status": "success",
  "predictions": [
    {
      "label": "person",
      "confidence": 0.96,
      "bbox": {
        "x": 249,
        "y": 16,
        "width": 391,
        "height": 434
      },
      "emoji": "🧑"
    },
    {
      "label": "person",
      "confidence": 0.855,
      "bbox": {
        "x": 444,
        "y": 7,
        "width": 196,
        "height": 191
      },
      "emoji": "🧑"
    },
    {
      "label": "teddy bear",
      "confidence": 0.952,
      "bbox": {
        "x": 259,
        "y": 159,
        "width": 116,
        "height": 120
      },
      "emoji": "🧸"
    }
  ],
  "metadata": {
    "processing_time": 0.303,
    "model_info": {
      "framework": "Ultralytics YOLOv8"
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
# Start yolo_oi7 service

# Activate virtual environment
source yolo_venv/bin/activate

# Start service
python REST.py
```

### Systemd Service
```bash
# Install service file
sudo cp services/yolo-api.service /etc/systemd/system/

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable yolo-api.service
sudo systemctl start yolo-api.service

# Check status
sudo systemctl status yolo-api.service
```

## Performance Optimization

### Hardware Requirements

**Minimum:**
- 4GB RAM
- 2-core CPU
- 10GB disk space

**Recommended:**
- 8GB+ RAM
- 4-core CPU
- RTX 3060 or better GPU
- NVMe SSD storage

### Model Performance

| Model | Size | Speed | mAP | RAM Usage | Use Case |
|-------|------|-------|-----|-----------|----------|
| YOLOv8n | 6MB | Very Fast | 37.3 | 1GB | Real-time apps |
| YOLOv8s | 22MB | Fast | 44.9 | 2GB | General purpose |
| YOLOv8m | 52MB | Medium | 50.2 | 4GB | Balanced accuracy |
| YOLOv8l | 87MB | Slow | 52.9 | 6GB | High accuracy |
| YOLOv8x | 136MB | Very Slow | 53.9 | 8GB | Maximum accuracy |

### Optimization Settings

- **Confidence Threshold**: 0.25 (filters low-confidence detections)
- **IoU Threshold**: 0.3 (removes overlapping detections)
- **GPU Acceleration**: Automatically detected and enabled
- **Batch Processing**: Single image optimized for API responses

## Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `Model not loaded` | YOLOv8 model failed to load | Check model files and GPU drivers |
| `File not found` | Invalid file path | Verify file exists and path is correct |
| `File too large` | Image exceeds MAX_FILE_SIZE | Raise MAX_FILE_SIZE or use a smaller file |
| `Invalid URL format` | Malformed image URL | Check URL syntax and accessibility |
| `URL does not point to an image` | Non-image content type | Ensure URL serves image content |

### Error Response Format
```json
{
  "service": "yolo",
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
response = requests.get('http://localhost:7773/analyze', 
                       params={'url': 'https://example.com/image.jpg'})
data = response.json()

# Object detection from file
response = requests.get('http://localhost:7773/analyze',
                       params={'file': '/path/to/image.jpg'})
result = response.json()

# Process results
for prediction in result['predictions']:
    label = prediction['label']
    confidence = prediction['confidence']
    bbox = prediction['bbox']
    emoji = prediction.get('emoji', '')
    print(f"{emoji} {label}: {confidence:.2f} at ({bbox['x']}, {bbox['y']})")
```

### JavaScript
```javascript
// Object detection from URL
const response = await fetch('http://localhost:7773/analyze?' + 
    new URLSearchParams({url: 'https://example.com/image.jpg'}));
const data = await response.json();

// Process detections
data.predictions.forEach(prediction => {
    console.log(`${prediction.emoji || ''} ${prediction.label}: ${prediction.confidence}`);
    console.log(`Location: (${prediction.bbox.x}, ${prediction.bbox.y})`);
    console.log(`Size: ${prediction.bbox.width}x${prediction.bbox.height}`);
});
```

## Troubleshooting

### Installation Issues
- **CUDA not detected**: Install NVIDIA drivers and CUDA toolkit
- **Model download fails**: Check internet connection and disk space
- **Dependencies missing**: Use requirements.txt in virtual environment

### Runtime Issues
- **Slow inference**: Ensure GPU is detected, use smaller model for speed
- **Memory errors**: Use smaller model or increase system RAM
- **Connection refused**: Verify service is running on correct port

### Performance Issues
- **Low detection accuracy**: Use larger model (yolov8l or yolov8x)
- **Too many false positives**: Increase confidence threshold
- **Missing objects**: Decrease confidence threshold or use larger model

## Security Considerations

### Access Control
- Set `PRIVATE=True` for localhost-only access
- Use reverse proxy (nginx) for production deployment
- Implement rate limiting for public endpoints

### File Security
- File uploads validated and cleaned up automatically
- Size limits prevent resource exhaustion
- Only allowed image formats accepted

## Supported Object Classes

YOLOv8 detects 80 COCO dataset classes:

**People & Animals:**
person, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe

**Vehicles:**
bicycle, car, motorcycle, airplane, bus, train, truck, boat

**Sports & Recreation:**
frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket

**Food & Dining:**
bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake

**Household Items:**
chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush

**Outdoor Objects:**
traffic light, fire hydrant, stop sign, parking meter, bench, backpack, umbrella, handbag, tie, suitcase

---

*Generated with Animal Farm ML Platform v3.0 - Open Images v7 Object Detection*
