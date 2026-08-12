FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 --index-url https://download.pytorch.org/whl/cu118

RUN pip install git+https://github.com/openai/CLIP.git

RUN pip install \
    numpy==2.2.6 \
    pillow==11.3.0

RUN pip install \
    flask==3.1.1 \
    flask-cors==6.0.1 \
    python-dotenv==1.1.1 \
    requests==2.32.4

COPY REST.py /app/

EXPOSE 7798

ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX"

CMD ["python", "REST.py"]
