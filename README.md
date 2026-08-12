# Animal Farm Model Zoo

This repository contains optional Animal Farm-compatible model services that are not part of the production Animal Farm bundle. The production `animal-farm` repo is kept smaller so users only receive the services currently used in production. This repo preserves alternate models and experimental or specialized services for users who want to run a different model behind the same standard REST shape.

Each service is self-contained in its own directory with a `REST.py` API, dependency list, environment sample, and usually a startup script and Dockerfile. Services can be run independently or combined with an Animal Farm deployment as long as their ports and downstream service configuration are aligned.

## What is Included

| Service | Default port | Purpose |
|---------|--------------|---------|
| `BLIP` | `7777` | BLIP image captioning service. |
| `LAVIS` | `7777` | BLIP2/LAVIS image-language service based on Salesforce LAVIS. |
| `CLIP` | `7778` | CLIP image-text similarity, label matching, and caption scoring. |
| `clip-score` | `7776` | CLIP caption/image scoring plus image and text embedding endpoints. |
| `CLIP_detection` | `7788` | Two-stage detector using YOLO segmentation proposals classified with CLIP. |
| `detectron2` | `7771` | Detectron2 COCO object detection and instance segmentation. |
| `hailo-YOLO` | `7773` | YOLO object detection intended for Hailo accelerator deployments. |
| `ollama-api` | `7782` | Ollama-backed text and vision model API for multimodal analysis. |
| `rtdetr2` | `7781` | RT-DETRv2 transformer-based COCO object detection. |
| `rtmdet` | `7792` | OpenMMLab RTMDet COCO object detection. |
| `speciesnet` | `7794` | SpeciesNet wildlife/camera-trap classifier with optional geolocation hints. |
| `xception` | `7779` | Xception ImageNet classifier. |
| `xception_detection` | `7799` | Two-stage detector using YOLO proposals classified with Xception. |
| `yolo_365` | `7773` | YOLO model trained for the Objects365 label set. |
| `yolo_oi7` | `7791` | YOLO model trained for the Open Images v7 label set. |

Ports come from the service `.env.sample` files unless otherwise noted in the service README. Several optional services share historical defaults, so change `PORT` in each service `.env` before running them together.

## Standard API Shape

Most services expose the same Animal Farm-compatible endpoint:

```bash
# Local file path
curl "http://localhost:<port>/analyze?file=/path/to/image.jpg" | jq

# Remote image URL
curl "http://localhost:<port>/analyze?url=https://example.com/image.jpg" | jq

# Multipart upload
curl -X POST -F "file=@/path/to/image.jpg" "http://localhost:<port>/analyze" | jq
```

Common supporting endpoints:

```bash
curl "http://localhost:<port>/health" | jq
curl "http://localhost:<port>/classes" | jq
```

`/classes` is available on detector/classifier services that have a fixed class list. `clip-score` is the main exception to the `/analyze` pattern: it exposes `/score`, `/embed/image`, and `/embed/text` for CLIP scoring and embeddings.

## Quick Start

Each service is installed and run from its own directory:

```bash
cd <service-directory>
cp .env.sample .env
python3 -m venv <service-name>_venv
source <service-name>_venv/bin/activate
pip install -r requirements.txt
./<service-start-script>.sh
```

Use the service README for exact setup details. Some services require CUDA-specific packages, model weights, external runtimes, or hardware-specific SDKs:

- `detectron2`, `rtmdet`, and `rtdetr2` have framework-specific installation requirements.
- `hailo-YOLO` requires the Hailo runtime/SDK stack.
- `ollama-api` requires an Ollama server and configured text or vision models.
- `LAVIS` includes upstream LAVIS files and has its own installation flow.

## Docker

Many services include a `Dockerfile`, and `rtmdet` also includes `docker-compose.yaml`.

```bash
cd <service-directory>
docker build -t animal-farm-<service> .
docker run --rm --gpus all --env-file .env -p <port>:<port> animal-farm-<service>
```

Adjust GPU flags, volume mounts, and model-cache paths for your host. Some model weights are large and may need to be mounted or downloaded before the container starts.

## Using a Model with Animal Farm

To swap one of these services into an Animal Farm deployment:

1. Start the model service and verify `/health`.
2. Confirm the service returns the expected response from `/analyze` or its documented endpoint.
3. Point the Animal Farm orchestrator or downstream caller at the service host and port.
4. Keep only one service per port, or override `PORT` in `.env`.
5. Review the service README for response details, class lists, thresholds, and model configuration.

The goal is compatibility at the service boundary, not identical model behavior. Different label sets, confidence thresholds, segmentation support, and captioning models will produce different outputs even when the JSON envelope follows the same standard.

## Repository Layout

The expected service structure is:

```text
service-name/
  README.md
  REST.py
  .env.sample
  requirements.txt
  <startup-script>.sh
  Dockerfile
```

Some directories include additional model configs, label maps, weights, test scripts, or framework-specific files.

## License

This repository is licensed under the GNU General Public License v3.0. See `LICENSE` for details. Individual model weights and upstream frameworks may have their own licenses and usage restrictions.

Note: the initial repository metadata briefly listed MIT by mistake; this repository is GPL-3.0 to match Animal Farm.
