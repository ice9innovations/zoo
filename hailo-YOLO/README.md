# HAILO-Accelerated YOLO Object Detection Service

**Platform**: Raspberry Pi 5 with HAILO-8L M.2 AI Accelerator
**Framework**: HAILO Platform SDK + YOLOv8s compiled to HEF
**Purpose**: Real-time object detection with bounding boxes and emoji mapping
**Status**: Active

> **Not a server service.** This runs on a Raspberry Pi 5 with a HAILO-8L M.2 HAT, not on the main Animal Farm inference server. The `hailo_platform` package is hardware-specific and will not install on standard x86 machines.

## Overview

HAILO-Accelerated YOLO offloads YOLOv8 inference to the HAILO-8L Neural Processing Unit (NPU), which delivers ~13 TOPS at a fraction of the power consumption of a GPU. The model runs as a compiled `.hef` (HAILO Executable Format) file — there are no PyTorch tensors, CUDA, or Ultralytics dependencies at inference time.

The service detects 80 COCO object classes with bounding boxes and emoji mapping, matching the output format of the server-side YOLOv8 service.

## Files

| File | Description |
|------|-------------|
| `REST.py` | Service implementation — in-memory pipeline, letterboxing, no disk I/O |
| `yolo-api.service` | systemd unit (runs as `hailo` user at `/home/hailo/hailo-YOLO`) |
| `yolo.sh` | Startup script |
| `requirements.txt` | Python dependencies (excludes `hailo_platform` — system package) |

## Hardware Requirements

- Raspberry Pi 5 (4GB or 8GB RAM)
- HAILO-8L M.2 AI Accelerator (M.2 HAT+ or compatible carrier)
- Raspberry Pi OS (Bookworm or later)
- HAILO drivers and platform SDK installed system-wide

## Installation

### 1. Install HAILO Platform (system-wide)

```bash
sudo apt update
sudo apt install python3-hailo-platform hailo-firmware
```

The HAILO SDK is not available on PyPI. It must be installed via apt from the HAILO repository. See the official HAILO documentation for repository setup.

### 2. Install Model File

```bash
# The compiled HEF model ships with the HAILO model zoo package
sudo apt install hailo-models
# Model is installed at:
ls /usr/share/hailo-models/yolov8s_h8l.hef
```

### 3. Environment Setup

```bash
cd /home/hailo/hailo-YOLO

# Create virtual environment with system site packages
# (required to access hailo_platform installed via apt)
python3 -m venv venv --system-site-packages

source venv/bin/activate
pip install -r requirements.txt
```

### 4. Configuration

Create a `.env` file:

```bash
PORT=7773
PRIVATE=false
AUTO_UPDATE=true
```

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PORT` | Yes | — | Service listening port |
| `PRIVATE` | Yes | — | `true` = localhost only, `false` = all interfaces |
| `AUTO_UPDATE` | No | `true` | Fetch emoji mappings from GitHub on startup |

## API Endpoints

### Health Check
```
GET /health
```

```json
{
  "status": "healthy",
  "model_status": "loaded",
  "model_info": {
    "device": "HAILO-8L Hardware Acceleration",
    "framework": "HAILO Platform"
  },
  "confidence_threshold": 0.25,
  "iou_threshold": 0.3,
  "supported_classes": 80
}
```

### Analyze Image
```
GET /analyze?url=<image_url>
GET /analyze?file=<file_path>
POST /analyze  (multipart file upload)
```

Exactly one of `url`, `file`, or POST body must be provided.

```json
{
  "service": "hailo_yolo",
  "status": "success",
  "predictions": [
    {
      "label": "dog",
      "confidence": 0.918,
      "bbox": { "x": 120, "y": 45, "width": 310, "height": 280 },
      "emoji": "🐕"
    }
  ],
  "metadata": {
    "processing_time": 0.041,
    "model_info": { "framework": "HAILO-8L Hardware Acceleration" }
  }
}
```

### Other Endpoints
```
GET /classes     # List all 80 COCO class names
```

## Service Management

```bash
# Install
sudo cp yolo-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable yolo-api.service
sudo systemctl start yolo-api.service

# Status
systemctl status yolo-api.service
```

The service runs as the `hailo` user with `WorkingDirectory=/home/hailo/hailo-YOLO`.

## How HAILO Inference Works

Unlike server-side services that use PyTorch directly, HAILO inference goes through the platform SDK:

1. Image is letterboxed to 640×640 (aspect-ratio preserving, gray padding)
2. Input is passed to `InferVStreams` as a `uint8` numpy array
3. Output is a class-wise list of 80 arrays (one per COCO class), each containing `[x1, y1, x2, y2, confidence]` in normalized coordinates
4. Bounding boxes are mapped back to original image coordinates using the letterbox scale and padding offsets
5. IoU filtering removes overlapping detections per class

The model file (`yolov8s_h8l.hef`) is a YOLOv8s model compiled by HAILO's DFC (Dataflow Compiler) for the HAILO-8L architecture. It cannot run on any other hardware.

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| `ModuleNotFoundError: hailo_platform` | SDK not installed or venv missing `--system-site-packages` | Reinstall venv with `--system-site-packages` |
| `FileNotFoundError: yolov8s_h8l.hef` | HAILO model zoo not installed | `sudo apt install hailo-models` |
| `VDevice()` fails | HAILO device not found or driver not loaded | Check `dmesg` for HAILO PCIe device; verify M.2 HAT seating |
| Empty detections | Inference silent failure | Check logs for HAILO output format warnings |

## Supported Classes

80 COCO classes — identical to the server-side YOLOv8 service. See `REST.py` for the full `COCO_CLASSES` list.
