#!/bin/bash
cd "$(dirname "$0")"
source CLIP_venv/bin/activate
source .env
#exec env CUDA_LAUNCH_BLOCKING=1 python3 REST.py
python REST.py
