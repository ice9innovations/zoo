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
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch first (with CUDA 11.8 support)
RUN pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install CLIP from GitHub (avoids ancient PyTorch dependency)
RUN pip install git+https://github.com/openai/CLIP.git

# Install other ML dependencies
RUN pip install \
    "numpy==2.2.6" \
    pillow==11.3.0 \
    scipy

# Install web service dependencies
RUN pip install \
    Flask==3.1.2 \
    flask-cors==6.0.1 \
    python-dotenv==1.1.1 \
    requests==2.32.5

# Create labels directory
RUN mkdir -p /app/labels

# Copy CLIP service files
COPY REST.py /app/
COPY clip_analyzer.py /app/
COPY .env /app/
COPY labels/ /app/labels/

# Expose port
EXPOSE 7778

# Set environment variables
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX"

# Run the service
CMD ["python", "REST.py"]
