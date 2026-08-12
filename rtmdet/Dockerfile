# RTMDet Object Detection Service
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster package installation
RUN pip install uv

# Copy requirements and install dependencies in exact order
COPY requirements.txt .

# Install core PyTorch dependencies first (critical order)
# Use CUDA-enabled versions that match working installation
RUN uv pip install --system torch==2.1.2+cu121 torchvision==0.16.2+cu121 --index-url https://download.pytorch.org/whl/cu121
RUN uv pip install --system numpy==1.26.4

# Install OpenMMLab installer
RUN uv pip install --system openmim

# Install MMDetection ecosystem via MIM (critical order)
RUN mim install mmengine
RUN mim install "mmcv==2.1.0"
RUN mim install mmdet

# Install Flask and other dependencies
RUN uv pip install --system flask flask-cors python-dotenv Pillow requests opencv-python==4.11.0

# Copy application code
COPY . .

# Create uploads directory
RUN mkdir -p uploads

# Set environment variables with defaults
ENV PRIVATE=false
ENV PORT=7792
ENV CONFIDENCE_THRESHOLD=0.25
ENV AUTO_UPDATE=true
ENV TIMEOUT=10.0
ENV PYTHONPATH=/app

# Expose port
EXPOSE 7792

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7792/health || exit 1

# Run the application
CMD ["python", "REST.py"]