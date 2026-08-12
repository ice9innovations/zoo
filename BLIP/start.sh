#!/usr/bin/env bash
set -euo pipefail

MODEL_FILE="/usr/src/app/model_base_capfilt_large.pth"

echo "[BLIP] Starting BLIP service initialization..."

if [[ ! -f "$MODEL_FILE" ]];
then
  if [[ -n "${MODEL_URL:-}" ]];
  then
    echo "[BLIP] Model not found. Downloading from MODEL_URL: $MODEL_URL"
    curl -fL --progress-bar "$MODEL_URL" -o "$MODEL_FILE"
    echo "[BLIP] Model downloaded to $MODEL_FILE"
  else
    echo "[BLIP] ERROR: Model file not present and MODEL_URL not provided."
    echo "[BLIP] Please set MODEL_URL to a reachable URL for model_base_capfilt_large.pth, or mount the file."
    exit 1
  fi
else
  echo "[BLIP] Found existing model file at $MODEL_FILE"
fi

mkdir -p /usr/src/app/uploads

echo "[BLIP] Launching service..."
exec python3 /usr/src/app/REST.py


