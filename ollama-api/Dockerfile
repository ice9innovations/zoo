FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=0 \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/lib:$LD_LIBRARY_PATH \
    PATH=/usr/local/cuda/bin:$PATH

# Install system dependencies including CUDA runtime libs and Ollama
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev curl wget \
    libnvjpeg11 \
    webp \
    libnvtoolsext1 \
 && curl -fsSL https://ollama.com/install.sh | sh \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=7782 \
    PRIVATE=false \
    API_HOST=api \
    API_PORT=8080 \
    API_TIMEOUT=10

EXPOSE 7782

# Create startup script
RUN echo '#!/bin/bash\n\
# Start Ollama server in background\n\
ollama serve &\n\
\n\
# Wait for Ollama to start\n\
echo "Waiting for Ollama to start..."\n\
while ! curl -s http://localhost:11434/api/tags > /dev/null; do\n\
    sleep 1\n\
done\n\
echo "Ollama started successfully"\n\
\n\
# Start the Flask API (which handles model downloads)\n\
python3 REST.py\n\
' > /usr/src/app/start.sh && chmod +x /usr/src/app/start.sh

CMD ["/usr/src/app/start.sh"]

