#!/bin/bash
#cd "$(dirname "$0")"

cd "/home/hailo/hailo-rpi5-examples"
source setup_env.sh

cd "/home/hailo/hailo-YOLO"
source .env
python3 REST.py
