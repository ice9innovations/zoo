#!/bin/bash
cd "$(dirname "$0")"
source detectron2_venv/bin/activate
source .env
python3 REST.py
