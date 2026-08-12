FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for detectron2
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
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch first (with CUDA 11.8 support)
RUN pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install detectron2 dependencies
RUN pip install \
    opencv-python-headless==4.12.0.88 \
    pillow==11.3.0 \
    numpy==2.2.6 \
    scipy \
    matplotlib

# Clone detectron2 repo and install
RUN git clone https://github.com/facebookresearch/detectron2.git /app/detectron2_repo
RUN pip install -e /app/detectron2_repo

# Copy demo files (including predictor.py) to working directory
RUN cp /app/detectron2_repo/demo/predictor.py /app/

# Copy configs directory from detectron2 repo
RUN cp -r /app/detectron2_repo/configs /app/configs

# Copy COCO classes file
COPY coco_classes.txt /app/

# Install web service dependencies
RUN pip install \
    Flask==3.1.2 \
    flask-cors==6.0.1 \
    python-dotenv==1.1.1 \
    requests==2.32.5

# Copy detectron2 service files
COPY REST.py /app/
COPY detectron2_analyzer.py /app/
COPY .env /app/

# Expose port
EXPOSE 7771

# Set environment variables
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX"

# Run the service
CMD ["python", "REST.py"]