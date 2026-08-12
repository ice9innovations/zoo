#!/bin/bash
# Startup script for YOLO11n-Xception Detector Service

echo "Starting YOLO11n-Xception Detector Service..."

# Activate the virtual environment
source xception_venv/bin/activate

# Basic CUDA settings - minimal to avoid conflicts
#export CUDA_LAUNCH_BLOCKING=1
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Start the service with venv Python
python REST.py
