#!/bin/bash
cd "$(dirname "$0")"
source xception_venv/bin/activate
source .env
export LD_LIBRARY_PATH="$(dirname "$0")/xception_venv/lib/python3.9/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
python3 REST.py
