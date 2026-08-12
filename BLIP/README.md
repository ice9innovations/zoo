# BLIP Image Captioning Service

**Port**: 7777  
**Framework**: Salesforce BLIP (Bootstrapping Language-Image Pre-training)  
**Purpose**: AI-powered image captioning with emoji mapping  
**Status**: ✅ Active

## Overview

BLIP provides state-of-the-art image captioning using Salesforce's BLIP model. The service analyzes images and generates natural language descriptions, automatically mapping words to relevant emojis for enhanced user experience.

## Features

- **Modern V3 API**: Clean, unified endpoint with intuitive parameters
- **Unified Input Handling**: Single endpoint for both URL and file path analysis
- **Emoji Integration**: Automatic word-to-emoji mapping using local dictionary
- **Model Flexibility**: Support for multiple BLIP model sizes
- **Security**: File validation, size limits, secure cleanup
- **Performance**: GPU acceleration with CUDA/MPS support

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.2+ (for GPU acceleration)
- 8GB+ RAM (16GB+ recommended for large models)

### 1. Environment Setup

```bash
# Navigate to BLIP directory
cd /home/sd/animal-farm/BLIP

# Create virtual environment
python3 -m venv blip_venv

# Activate virtual environment
source blip_venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Model Download

Download the BLIP model files:

```bash
# Download base model (recommended, 113MB)
wget https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth

# Download large filtered model (highest quality, 447MB)
wget https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth
```

### 3. BLIP Repository Setup

Install the BLIP model implementation:

```bash
# Clone BLIP repository (temporary)
git clone https://github.com/salesforce/BLIP.git temp_blip

# Copy model files
cp -r temp_blip/models ./

# Clean up
rm -rf temp_blip
```

## Configuration

### Environment Variables (.env)

Create a `.env` file in the BLIP directory:

```bash
# Service Configuration
PORT=7777                    # Service port (default: 7777)
PRIVATE=False               # Access mode (False=public, True=localhost-only)

# Configuration Updates (GitHub-first pattern)
AUTO_UPDATE=true           # Refresh emoji/MWE config from GitHub on startup
TIMEOUT=10.0               # Timeout for remote config downloads (seconds)


```

### Configuration Details

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PORT` | Yes | - | Service listening port |
| `PRIVATE` | Yes | - | Access control (False=public, True=localhost-only) |
| `AUTO_UPDATE` | No | `true` | Refresh emoji/MWE config from GitHub on startup, then cache locally |
| `TIMEOUT` | No | `10.0` | Timeout for remote config downloads |

### Model Selection

The service uses `model_base_capfilt_large.pth` by default. Available models:

| Model | Size | Quality | Speed | Description |
|-------|------|---------|-------|-------------|
| `model_base_14M.pth` | 14MB | Low | Fastest | Quick captions |
| `model_base.pth` | 113MB | Good | Fast | Balanced performance |
| `model_large.pth` | 447MB | High | Slower | Better quality |
| `model_base_capfilt_large.pth` | 447MB | Highest | Slowest | Production quality |

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
  "device": "cuda:0"
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
curl "http://localhost:7777/analyze?url=https://example.com/image.jpg"
```

#### Analyze Image from File Path
```bash
GET /analyze?file=<file_path>
```

**Example:**
```bash
curl "http://localhost:7777/analyze?file=/path/to/image.jpg"
```

#### POST Request (File Upload)
```bash
POST /analyze
Content-Type: multipart/form-data
```

**Example:**
```bash
curl -X POST -F "file=@/path/to/image.jpg" http://localhost:7777/analyze
```

**Input Validation:**
- Exactly one parameter must be provided (`url` OR `file`)
- Cannot provide both parameters simultaneously
- Returns error if neither parameter is provided

**Response Format:**
```json
{
  "metadata": {
    "model_info": {
      "framework": "BLIP (Bootstrapping Language-Image Pre-training)"
    },
    "processing_time": 0.641
  },
  "predictions": [
    {
      "emoji_mappings": [
        { "emoji": "🧑", "word": "soldier" },
        { "emoji": "👩", "word": "girl" },
        { "emoji": "🌼", "word": "bouquet" }
      ],
      "text": "a soldier giving a little girl a bouquet"
    }
  ],
  "service": "blip",
  "status": "success"
}
```

## Service Management

### Manual Startup

```bash
# Start service
cd /home/sd/animal-farm/BLIP
./BLIP.sh
```

### Systemd Service

```bash
# Install service
sudo cp services/BLIP-api.service /etc/systemd/system/
sudo systemctl daemon-reload

# Start/stop service
sudo systemctl start BLIP-api
sudo systemctl stop BLIP-api

# Enable auto-start
sudo systemctl enable BLIP-api

# Check status
sudo systemctl status BLIP-api

# View logs
sudo journalctl -u BLIP-api -f
```

## Supported Formats

### Input Formats
- **Images**: PNG, JPG, JPEG, GIF, BMP, WebP
- **Max Size**: 32MB default
- **Input Methods**: URL, file upload, local path

### Output Features
- **Caption Generation**: Natural language descriptions
- **Emoji Mapping**: Automatic word-to-emoji conversion
- **Multi-word Expressions**: Support for compound terms
- **Confidence Scores**: Processing time metadata

## Performance Optimization

### Hardware Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|--------|
| GPU | GTX 1060 6GB | RTX 3080+ | CUDA 11.2+ required |
| RAM | 8GB | 16GB+ | Depends on model size |
| Storage | 2GB | 5GB+ | Models + temp files |

### Optimization Settings

```python
# Model precision (in REST.py)
use_fp16 = False  # Set to True for RTX cards with sufficient VRAM

# Batch processing
num_beams = 1     # Faster inference
max_length = 20   # Shorter captions for speed
```

## Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| "Model not loaded" | Missing model file | Download required model |
| "File too large" | File exceeds MAX_FILE_SIZE | Raise MAX_FILE_SIZE or use a smaller file |
| "Invalid URL" | Malformed URL | Check URL format |
| "Connection timeout" | Network issues | Check internet connection |
| "CUDA out of memory" | Insufficient VRAM | Use smaller model or reduce batch size |

### Error Response Format

```json
{
  "service": "blip",
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
    "http://localhost:7777/analyze",
    params={"url": "https://example.com/image.jpg"}
)

# File input
response = requests.get(
    "http://localhost:7777/analyze",
    params={"file": "/path/to/image.jpg"}
)

# POST file upload
with open('/path/to/image.jpg', 'rb') as f:
    response = requests.post(
        "http://localhost:7777/analyze",
        files={'file': f}
    )

result = response.json()
if result["status"] == "success":
    prediction = result["predictions"][0]
    text = prediction["text"]
    emoji_mappings = prediction["emoji_mappings"]
    emojis = [mapping["emoji"] for mapping in emoji_mappings]
    print(f"Caption: {text}")
    print(f"Emojis: {' '.join(emojis)}")
```

### JavaScript Integration

```javascript
// URL input
const response = await fetch(
  'http://localhost:7777/analyze?url=https://example.com/image.jpg'
);

// File input
const response = await fetch(
  'http://localhost:7777/analyze?file=/path/to/image.jpg'
);

// POST file upload
const formData = new FormData();
formData.append('file', fileInput.files[0]);
const response = await fetch('http://localhost:7777/analyze', {
  method: 'POST',
  body: formData
});

const result = await response.json();
if (result.status === 'success') {
  const prediction = result.predictions[0];
  console.log('Caption:', prediction.text);
  const emojis = prediction.emoji_mappings.map(mapping => mapping.emoji);
  console.log('Emojis:', emojis.join(' '));
}
```

## Troubleshooting

### Installation Issues

**Problem**: Import errors for BLIP models
```bash
# Solution: Ensure models directory exists
ls -la models/
# Should contain: blip.py, vit.py, etc.
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
netstat -tlnp | grep 7777

# Check logs
tail -f /var/log/syslog | grep BLIP
```

**Problem**: Emoji mapping failures
```bash
# Check cached config files
ls -la emoji_mappings.json mwe.txt

# Check config loading settings
grep -E 'AUTO_UPDATE|TIMEOUT' .env
```

### Performance Issues

**Problem**: Slow inference
- Use smaller model (`model_base.pth` instead of `model_base_capfilt_large.pth`)
- Enable FP16 precision for RTX cards
- Reduce `max_length` and use `num_beams=1`

**Problem**: Memory errors
- Reduce image size before processing
- Use CPU instead of GPU for small workloads
- Ensure sufficient swap space

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
