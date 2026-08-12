# RTMDet Installation Guide

## Tested Working Configuration

The RTMDet service requires specific versions of dependencies to work properly. Follow this exact installation sequence:

### 1. Clean Installation (if needed)
```bash
uv pip uninstall torch torchvision numpy openmim mmengine mmcv mmdet
```

### 2. Install Core Dependencies (in order)
```bash
# PyTorch ecosystem - CUDA-enabled versions required
uv pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 --index-url https://download.pytorch.org/whl/cu121
uv pip install numpy==1.26.4

# OpenMMLab installer
uv pip install openmim
```

### 3. Install MMDetection Ecosystem via MIM
```bash
# Install in this exact order
mim install mmengine
mim install "mmcv==2.1.0"
mim install mmdet
```

### 4. Install Additional Dependencies
```bash
uv pip install flask flask-cors python-dotenv Pillow requests opencv-python==4.11.0
```

## Environment Variables

Create a `.env` file in the rtmdet directory:
```bash
PRIVATE=false
PORT=7792
CONFIDENCE_THRESHOLD=0.25
AUTO_UPDATE=true
TIMEOUT=10.0
```

## Verification

Test that everything is working:
```bash
python -c "import torch; import mmdet; import mmcv; import mmengine; print('All dependencies loaded successfully')"
```

## Common Issues

### Dependency Hell
- **Problem**: Version conflicts between numpy, torch, and mmcv
- **Solution**: Follow the exact installation order above

### Model Loading Issues
- **Problem**: "checkpoint is None" warnings
- **Solution**: The service now downloads proper model checkpoints automatically

### CUDA Issues
- **Problem**: Model loads but gives poor results
- **Solution**: Ensure CUDA is properly configured with `torch.cuda.is_available()`

## Docker Installation (Recommended)

For a simpler installation that handles all dependencies automatically:

### Build and Run with Docker Compose
```bash
# Build and start the service
docker-compose up --build

# Run in background
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop the service
docker-compose down
```

### Build and Run with Docker
```bash
# Build the image
docker build -t rtmdet-service .

# Run the container
docker run -d \
  --name rtmdet \
  -p 7792:7792 \
  -e PRIVATE=false \
  -e PORT=7792 \
  -e CONFIDENCE_THRESHOLD=0.25 \
  rtmdet-service

# View logs
docker logs -f rtmdet

# Stop and remove
docker stop rtmdet && docker rm rtmdet
```

### GPU Support (Optional)
To enable GPU acceleration, uncomment the GPU lines in docker-compose.yml and ensure you have:
- NVIDIA Docker runtime installed
- Compatible GPU drivers

## Notes

- This configuration has been tested and works with the modernized RTMDet service
- The service follows the same patterns as detectron2 for consistency
- Models are downloaded automatically on first use
- Docker handles all the dependency complexity automatically