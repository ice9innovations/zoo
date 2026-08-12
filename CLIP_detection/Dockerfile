FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    ffmpeg \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch first (with CUDA 11.8 support)
RUN pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install YOLO and computer vision dependencies
RUN pip install \
    ultralytics \
    opencv-python-headless==4.12.0.88 \
    pillow==11.3.0 \
    numpy==2.2.6

# Install web service dependencies
RUN pip install \
    Flask==3.1.2 \
    flask-cors==6.0.1 \
    python-dotenv==1.1.1 \
    requests==2.32.5

# Copy CLIP_detection service files
COPY REST.py /app/
COPY two_stage_clip_detector.py /app/
copy yolo.py /app/
COPY .env /app/

# Expose port
EXPOSE 7788

# Set environment variables
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX"

# Run the service
CMD ["python", "REST.py"]
