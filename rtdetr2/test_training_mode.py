#!/usr/bin/env python3
"""
Test script to load RT-DETRv2 in training mode (separate postprocessor)
and compare results with deploy mode
"""

import sys
import os
sys.path.insert(0, '/home/sd/animal-farm/rtdetr2/RT-DETRv2/rtdetrv2_pytorch')

import torch
import torchvision.transforms as T
from PIL import Image
from src.core import YAMLConfig
from src.misc import dist_utils

def load_training_mode_model():
    """Load model and postprocessor separately like official evaluation"""
    # Use the same config as the official test
    config_path = '/home/sd/animal-farm/rtdetr2/RT-DETRv2/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r101vd_6x_coco.yml'
    resume_path = '/home/sd/.cache/torch/hub/checkpoints/rtdetrv2_r101vd_6x_coco_from_paddle.pth'

    # Setup like official evaluation
    update_dict = {'num_workers': 0, 'resume': resume_path, 'test_only': True}
    cfg = YAMLConfig(config_path, **update_dict)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = cfg.model.to(device)
    postprocessor = cfg.postprocessor.to(device)

    # Load weights
    checkpoint = torch.load(resume_path, map_location='cpu')
    if 'ema' in checkpoint:
        model.load_state_dict(checkpoint['ema']['module'])
        print("Loaded EMA weights")
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        print("Loaded model weights")
    else:
        print("Warning: No recognized weights found in checkpoint")

    model.eval()
    return model, postprocessor, device

def test_single_image():
    """Test single image with training mode"""
    print("Loading model in training mode...")
    model, postprocessor, device = load_training_mode_model()

    # Load test image
    import requests
    from io import BytesIO

    image_url = "http://k1.local/val2017/000000262487.webp"  # Baseball image
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert('RGB')
    w, h = image.size
    print(f"Image size: {w}x{h}")

    # Transform image (same as API)
    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    im_data = transforms(image)[None].to(device)

    # Original size for postprocessor
    orig_target_sizes = torch.tensor([[w, h]], device=device)

    print("Running inference in training mode...")
    with torch.no_grad():
        # Training mode: model outputs raw results
        outputs = model(im_data)
        print(f"Model outputs type: {type(outputs)}")
        print(f"Model outputs keys: {outputs.keys() if isinstance(outputs, dict) else 'Not a dict'}")

        # Training mode: separate postprocessor call
        results = postprocessor(outputs, orig_target_sizes)
        print(f"Postprocessor results type: {type(results)}")
        print(f"Number of results: {len(results)}")

        if len(results) > 0:
            result = results[0]  # First (and only) image
            print(f"Result keys: {result.keys()}")

            if 'boxes' in result:
                print(f"Number of detections: {len(result['boxes'])}")
                print(f"Boxes shape: {result['boxes'].shape}")
                print(f"Scores shape: {result['scores'].shape}")
                print(f"Labels shape: {result['labels'].shape}")

                # Show top 10 detections
                top_10_indices = torch.argsort(result['scores'], descending=True)[:10]
                for i, idx in enumerate(top_10_indices):
                    score = result['scores'][idx].item()
                    label = result['labels'][idx].item()
                    box = result['boxes'][idx]
                    print(f"  {i+1}. Label: {label}, Score: {score:.3f}, Box: {box}")

if __name__ == "__main__":
    test_single_image()