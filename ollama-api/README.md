# LLaMa (LLaVa) Service

**Port**: 7782  
**Framework**: Ollama (Meta LLaMA, Mistral, CodeLlama)  
**Purpose**: Large Language Model inference with vision and text capabilities  
**Status**: ✅ Active

## Overview

Ollama provides access to powerful open-source language models including LLaMA, Mistral, CodeLlama, and vision models like LLaVA. The service handles both text generation and image analysis through a unified API, with automatic emoji mapping for enhanced user experience.

## Features

- **Modern V3 API**: Clean, unified endpoint with intuitive parameters
- **Unified Input Handling**: Single endpoint for both URL and file path analysis
- **Multi-Modal Support**: Text generation and vision analysis capabilities
- **Model Flexibility**: Support for multiple Ollama models (text, vision, code)
- **Emoji Integration**: Automatic word-to-emoji mapping using local dictionary
- **Async Processing**: Non-blocking inference with proper resource management
- **Custom Prompts**: Configurable prompts and temperature settings
- **Security**: File validation, size limits, secure cleanup

## Installation

### Prerequisites

- Python 3.8+
- Ollama server running locally
- 16GB+ RAM (32GB+ recommended for large models)
- GPU with 8GB+ VRAM recommended for vision models

### 1. Environment Setup

```bash
# Navigate to Ollama API directory
cd /home/sd/animal-farm/ollama-api

# Create virtual environment
python3 -m venv ollama_venv

# Activate virtual environment
source ollama_venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Ollama Server Setup

Install and start Ollama server:

```bash
# Install Ollama (Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
ollama serve

# Pull recommended models
ollama pull llama2                    # Text generation
ollama pull llava-llama3             # Vision analysis
ollama pull mistral                  # Alternative text model
ollama pull codellama               # Code-focused model
```

### 3. Model Configuration

The service supports multiple model types:

```bash
# Text Models
ollama pull llama2:7b              # Default text model
ollama pull llama2:13b             # Larger text model
ollama pull mistral                # Fast inference
ollama pull codellama              # Code generation

# Vision Models  
ollama pull llava                  # Vision analysis
ollama pull llava:13b              # Larger vision model
ollama pull bakllava               # Alternative vision model

# Verify installation
ollama list
```

## Configuration

### Environment Variables (.env)

Create a `.env` file in the ollama-api directory:

```bash
# Service Configuration
PORT=7782                           # Service port
PRIVATE=False                       # Access mode (False=public, True=localhost-only)

# Configuration Updates (GitHub-first pattern)
AUTO_UPDATE=true                   # Refresh emoji/MWE config from GitHub on startup

# Ollama Configuration
OLLAMA_HOST=http://localhost:11434  # Ollama server URL
TEXT_MODEL=llama2                   # Default text model
VISION_MODEL=llava-llama3           # Default vision model
TEMPERATURE=0.3                     # Sampling temperature (0.0-2.0)

# Prompts
PROMPT=What is in this image? One sentence only.  # Default vision prompt
```

### Configuration Details

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PORT` | Yes | - | Service listening port |
| `PRIVATE` | Yes | - | Access control (False=public, True=localhost-only) |
| `AUTO_UPDATE` | No | `true` | Refresh emoji/MWE config from GitHub on startup, then cache locally |
| `OLLAMA_HOST` | Yes | - | Ollama server URL |
| `TEXT_MODEL` | Yes | - | Default model for text generation |
| `VISION_MODEL` | Yes | - | Default model for image analysis |
| `TEMPERATURE` | Yes | - | Sampling temperature for generation |
| `PROMPT` | Yes | - | Default prompt for vision analysis |

## API Endpoints

### Health Check
```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "ollama_host": "http://localhost:11434",
  "text_model": "llama2",
  "vision_model": "llava-llama3",
  "temperature": 0.3
}
```

### Available Models
```bash
GET /models
```

**Response:**
```json
{
  "status": "success",
  "models": {
    "text": ["llama2", "mistral", "codellama"],
    "vision": ["llava", "llava:13b"],
    "code": ["codellama", "deepseek-coder"]
  },
  "total_models": 6
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
- `prompt` (string, optional): Custom prompt for analysis
- `model` (string, optional): Specific model to use
- `temperature` (float, optional): Sampling temperature (0.0-2.0)

**Note:** Exactly one parameter (`url` or `file`) must be provided.

**Response:**
```json
{
  "service": "ollama",
  "status": "success",
  "predictions": [
    {
      "text": "A man is giving a teddy bear to a little girl who is wearing a pink dress.",
      "emoji_mappings": [
        {"word": "man", "emoji": "🧑"},
        {"word": "teddy_bear", "emoji": "🧸"},
        {"word": "girl", "emoji": "👩"},
        {"word": "dress", "emoji": "👗"}
      ]
    }
  ],
  "metadata": {
    "processing_time": 0.216,
    "model_info": {
      "framework": "Ollama",
      "model": "llava-llama3",
      "prompt": "Briefly describe this image in a single short sentence."
    }
  }
}
```

### V2 Compatibility Routes
```bash
GET /v2/analyze?image_url=<url>       # Translates to V3 url parameter
GET /v2/analyze_file?file_path=<path>  # Translates to V3 file parameter
```

### Text Generation
```bash
POST /text
```

**Request Body:**
```json
{
  "prompt": "Explain quantum computing in simple terms",
  "model": "llama2",
  "temperature": 0.7
}
```

**Response:**
```json
{
  "status": "success",
  "response": "Quantum computing uses quantum mechanical phenomena to process information differently than traditional computers...",
  "model_used": "llama2",
  "response_length": 256,
  "temperature": 0.7
}
```

### Image Analysis with Custom Prompt
```bash
POST /image
```

**Form Data:**
- `uploadedfile`: Image file
- `prompt`: Custom analysis prompt
- `model`: Specific vision model

## Service Management

### Manual Startup
```bash
# Ensure Ollama is running
ollama serve

# Activate virtual environment
source ollama_venv/bin/activate

# Start service
python REST.py
```

### Systemd Service
```bash
# Install service file
sudo cp services/ollama-api.service /etc/systemd/system/

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable ollama-api.service
sudo systemctl start ollama-api.service

# Check status
sudo systemctl status ollama-api.service
```

## Performance Optimization

### Hardware Requirements

**Minimum:**
- 16GB RAM
- 4-core CPU
- 50GB disk space for models

**Recommended:**
- 32GB+ RAM
- 8-core CPU
- RTX 4070 or better for GPU acceleration
- NVMe SSD storage

### Model Performance

| Model | RAM Usage | Speed | Quality | Use Case |
|-------|-----------|-------|---------|----------|
| llama2:7b | 4GB | Fast | Good | General text |
| llama2:13b | 8GB | Medium | Better | Complex text |
| mistral | 4GB | Very Fast | Good | Quick responses |
| llava | 6GB | Medium | Good | Vision analysis |
| codellama | 4GB | Fast | Good | Code generation |

### Optimization Settings

- **Temperature**: Lower (0.1-0.3) for focused responses, higher (0.7-1.0) for creative output
- **Context Length**: Adjust based on use case and available memory
- **Concurrent Requests**: Service handles async processing with proper resource management

## Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `Ollama not available` | Ollama server not running | Start Ollama with `ollama serve` |
| `Model not found` | Requested model not installed | Pull model with `ollama pull <model>` |
| `Temperature out of range` | Invalid temperature value | Use values between 0.0 and 2.0 |
| `Prompt too long` | Prompt exceeds 10,000 characters | Reduce prompt length |
| `File too large` | Image exceeds MAX_FILE_SIZE | Raise MAX_FILE_SIZE or use a smaller file |

### Error Response Format
```json
{
  "service": "ollama",
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

# Vision analysis
response = requests.get('http://localhost:7782/analyze', 
                       params={
                           'url': 'https://example.com/image.jpg',
                           'prompt': 'Describe this image in detail',
                           'model': 'llava'
                       })
data = response.json()

# Text generation
response = requests.post('http://localhost:7782/text',
                        json={
                            'prompt': 'Write a Python function to sort a list',
                            'model': 'codellama',
                            'temperature': 0.2
                        })
result = response.json()
```

### JavaScript
```javascript
// Vision analysis
const visionResponse = await fetch('http://localhost:7782/analyze', {
    method: 'GET',
    headers: {'Content-Type': 'application/json'},
    params: new URLSearchParams({
        url: 'https://example.com/image.jpg',
        prompt: 'What objects are in this image?'
    })
});

// Text generation
const textResponse = await fetch('http://localhost:7782/text', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        prompt: 'Explain machine learning concepts',
        model: 'llama2',
        temperature: 0.5
    })
});
```

## Troubleshooting

### Installation Issues
- **Ollama installation fails**: Use official installation script from ollama.com
- **Model download slow**: Ensure stable internet connection, models are large (GB)
- **Python dependencies**: Use virtual environment and install from requirements.txt

### Runtime Issues
- **Connection refused**: Verify Ollama server is running on correct port (11434)
- **Model loading slow**: First inference loads model into memory, subsequent calls faster
- **Out of memory**: Use smaller models or increase system RAM

### Performance Issues
- **Slow responses**: Use GPU acceleration, smaller models, or reduce temperature
- **High memory usage**: Monitor model memory usage with `ollama ps`
- **Timeouts**: Increase timeout values for complex prompts or large images

## Security Considerations

### Access Control
- Set `PRIVATE=True` for localhost-only access
- Use reverse proxy (nginx) for production deployment
- Implement rate limiting for public endpoints

### Model Security
- Models run locally, no external API calls for inference
- Prompts and responses stay on your infrastructure
- File uploads validated and cleaned up automatically

## Supported Models

The service supports various Ollama model categories:

### Text Models
- **LLaMA 2**: General-purpose conversational AI
- **Mistral**: Fast, efficient inference
- **CodeLlama**: Code generation and analysis
- **Dolphin**: Fine-tuned for instruction following

### Vision Models
- **LLaVA**: Large Language and Vision Assistant
- **BakLLaVA**: Alternative vision model
- **LLaVA-1.5**: Improved vision understanding

### Specialized Models
- **DeepSeek Coder**: Advanced code generation
- **Mixtral**: Mixture of experts architecture
- **Phi**: Microsoft's small language model

---

*Generated with Animal Farm ML Platform v3.0 - Ollama Integration*
