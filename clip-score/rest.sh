#!/bin/bash
cd "$(dirname "$0")"
source /home/sd/CLIP/clip_venv/bin/activate
source .env
python3 REST.py
