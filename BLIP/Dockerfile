# CUDA-enabled PyTorch base matching torch==2.5.1+cu121
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

# Preinstall torch/torchvision matching cu121 to avoid lengthy builds
RUN pip3 install --no-cache-dir \
    torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x start.sh

ENV PORT=7777 \
    PRIVATE=false \
    API_HOST=api \
    API_PORT=8080 \
    API_TIMEOUT=10

EXPOSE 7777

CMD ["/usr/src/app/start.sh"]

