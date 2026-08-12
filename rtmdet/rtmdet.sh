#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
source .env
python3 REST.py
