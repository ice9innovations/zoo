# CLIP Caption Scoring Service

**Port**: 7776
**Framework**: OpenAI CLIP (ViT-L/14)
**Purpose**: Cosine similarity scoring between an image and a caption string
**Status**: ✅ Active

## Overview

clip-score computes how well a caption describes an image using CLIP's cosine similarity between image and text embeddings. It is a focused extraction of the scoring logic from the CLIP classification service — no label files, no embedding cache, no classification overhead.

Typical use: pass a VLM-generated caption (e.g. from claude-api, llama-cpp, or moondream) and the source image to get an objective quality score between 0 and 1.

## Installation

### Prerequisites

- Python 3.10+
- CUDA 12.1+ (for GPU acceleration)
- CLIP installed at `/home/sd/CLIP/clip` (symlinked into service directory)

### Setup

```bash
cd /home/sd/animal-farm/clip-score

# Symlink the CLIP package (already done)
ln -s /home/sd/CLIP/clip ./clip

# Create .env from sample
cp .env.sample .env

# Use the shared CLIP venv (already contains torch, clip-by-openai, flask)
# rest.sh points to /home/sd/CLIP/clip_venv automatically
```

## Configuration

### Environment Variables (.env)

```bash
PORT=7776
PRIVATE=False
CLIP_MODEL=ViT-L/14
```

### Configuration Details

| Variable | Required | Description |
|----------|----------|-------------|
| `PORT` | Yes | Service listening port |
| `PRIVATE` | Yes | `False` = all interfaces, `True` = localhost only |
| `CLIP_MODEL` | Yes | CLIP model variant — must match the CLIP classification service for comparable scores |

### Model Options

| Model | VRAM | Notes |
|-------|------|-------|
| `ViT-B/32` | ~4GB | Faster, less accurate |
| `ViT-L/14` | ~8GB (4GB FP16) | Production default |
| `ViT-L/14@336px` | ~10GB | Higher resolution input |

## API Endpoints

### Health Check

```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "clip-score",
  "model": "ViT-L/14",
  "device": "cuda"
}
```

### Score Caption

```bash
GET /score?caption=<text>&url=<image_url>
GET /score?caption=<text>&file=<file_path>
POST /score  (multipart: caption + file)
```

#### URL Example
```bash
curl "http://localhost:7776/score?caption=a+dog+in+the+park&url=http://example.com/image.jpg"
```

#### File Path Example
```bash
curl "http://localhost:7776/score?caption=a+dog+in+the+park&file=/path/to/image.jpg"
```

#### Multipart Upload Example
```bash
curl -X POST \
  -F "caption=a dog in the park" \
  -F "file=@/path/to/image.jpg" \
  http://localhost:7776/score
```

**Input Validation:**
- `caption` is required and must be non-empty
- Exactly one image source must be provided (`url`, `file`, or multipart upload)
- Cannot provide both `url` and `file`

**Response Format:**
```json
{
  "service": "clip-score",
  "status": "success",
  "similarity_score": 0.2814,
  "caption": "a dog in the park",
  "image_embedding": [0.012, -0.034, ...],
  "text_embedding": [0.021, 0.008, ...],
  "metadata": {
    "processing_time": 0.183,
    "model_info": {
      "framework": "openai-clip",
      "model": "ViT-L/14",
      "device": "cuda"
    }
  }
}
```

**Score interpretation**: Scores typically range from 0.15 (poor match) to 0.35 (strong match). Scores above 0.28 indicate a good caption.

**Error Response:**
```json
{
  "service": "clip-score",
  "status": "error",
  "similarity_score": null,
  "error": {"message": "Must provide non-empty 'caption' parameter"},
  "metadata": {"processing_time": 0.001}
}
```

### Get Image Embedding

```bash
POST /embed/image
```

Returns the normalized CLIP image embedding without requiring a caption. Use this when you need the embedding for vector search or storage and don't need a similarity score.


#### URL Example
```bash
curl -X POST http://localhost:7776/embed/image \
  -H "Content-Type: application/json" \
  -d '{"url": "http://example.com/image.jpg"}'
```

#### File Path Example
```bash
curl -X POST http://localhost:7776/embed/image \
  -H "Content-Type: application/json" \
  -d '{"file": "/path/to/image.jpg"}'
```

#### Multipart Upload Example
```bash
curl -X POST \
  -F "file=@/path/to/image.jpg" \
  http://localhost:7776/embed/image
```

**Input Validation:**
- Exactly one image source must be provided (`url`, `file`, or multipart upload)
- Cannot provide both `url` and `file`

**Response Format:**
```json
{
  "service": "clip-score",
  "status": "success",
  "image_embedding": [0.012, -0.034, ...],
  "metadata": {
    "processing_time": 0.042,
    "model_info": {
      "framework": "openai-clip",
      "model": "ViT-L/14",
      "device": "cuda"
    }
  }
}
```

### Batch Text Embeddings

```bash
POST /embed/text
```

Returns CLIP text embeddings for a batch of terms (max 500). No image required. Embeddings are L2-normalized and identical to the `text_embedding` returned by `/score`.

> `/v3/embed/text` is deprecated but remains functional.

```bash
curl -X POST http://localhost:7776/embed/text \
  -H "Content-Type: application/json" \
  -d '{"terms": ["dog", "cat", "truck"]}'
```

**Response Format:**
```json
{
  "service": "clip-score",
  "status": "success",
  "embeddings": {
    "dog": [0.021, -0.003, ...],
    "cat": [0.018, 0.011, ...],
    "truck": [-0.007, 0.034, ...]
  },
  "metadata": {
    "processing_time": 0.031,
    "term_count": 3,
    "model_info": {
      "framework": "openai-clip",
      "model": "ViT-L/14",
      "device": "cuda"
    }
  }
}
```

---

## Service Management

### Manual Startup

```bash
cd /home/sd/animal-farm/clip-score
./rest.sh
```

### Systemd Service

```bash
# Install
sudo cp services/clip-score.service /etc/systemd/system/
sudo systemctl daemon-reload

# Start / stop
sudo systemctl start clip-score
sudo systemctl stop clip-score

# Enable auto-start on boot
sudo systemctl enable clip-score

# Check status
sudo systemctl status clip-score

# View logs
sudo journalctl -u clip-score -f
```

## Integration Examples

### Python

```python
import requests

# Score a caption against a URL
response = requests.get(
    "http://localhost:7776/score",
    params={
        "caption": "a man walking a dog",
        "url": "https://example.com/image.jpg"
    }
)
result = response.json()
print(result["similarity_score"])  # e.g. 0.2814

# Get image embedding only
response = requests.post(
    "http://localhost:7776/embed/image",
    json={"url": "https://example.com/image.jpg"}
)
result = response.json()
print(len(result["image_embedding"]))  # e.g. 768 for ViT-L/14

# Batch text embeddings
response = requests.post(
    "http://localhost:7776/embed/text",
    json={"terms": ["dog", "cat", "truck"]}
)
result = response.json()
print(result["embeddings"]["dog"])
```

### JavaScript

```javascript
// URL-based scoring
const response = await fetch(
  'http://localhost:7776/score?caption=a+man+walking+a+dog&url=https://example.com/image.jpg'
);
const result = await response.json();
console.log(result.similarity_score);

// Get image embedding only
const response = await fetch('http://localhost:7776/embed/image', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({url: 'https://example.com/image.jpg'})
});
const result = await response.json();
console.log(result.image_embedding.length);
```

## Troubleshooting

**Problem**: `ModuleNotFoundError: No module named 'clip'`
The `clip` symlink is missing. Run:
```bash
ln -s /home/sd/CLIP/clip /home/sd/animal-farm/clip-score/clip
```

**Problem**: Service fails to start — port already in use
```bash
lsof -ti :7776 | xargs kill
```

**Problem**: CUDA out of memory
Switch to `ViT-B/32` in `.env` — it uses roughly half the VRAM.

**Problem**: Low scores on clearly correct captions
Ensure `CLIP_MODEL` matches the model used when scores were originally calibrated. Scores are not comparable across model variants.

## Docker

```bash
# Build
docker build -t clip-score /home/sd/animal-farm/clip-score/

# Run
docker run -d \
  --name clip-score \
  --gpus all \
  --env-file /home/sd/animal-farm/clip-score/.env \
  -p 7798:7798 \
  clip-score

# Test
curl -s http://localhost:7798/health | python3 -m json.tool
curl -s "http://localhost:7798/score?caption=a+dog+in+the+park&url=https://example.com/image.jpg" | python3 -m json.tool

# Delete
docker stop clip-score && docker rm clip-score && docker rmi clip-score
```

> CLIP model is downloaded from OpenAI's GitHub at build time. No volume mount required.

---

**Last Updated**: 2026-02-20
**Service Version**: Production
