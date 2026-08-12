#!/usr/bin/env python3
"""
Test script to check COCO evaluation parameter differences
"""

import sys
import os
sys.path.append('/home/sd/animal-farm/benchmark')

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import tempfile
import requests
from io import BytesIO
from PIL import Image
import numpy as np

# Import our training mode test
sys.path.insert(0, '/home/sd/animal-farm/rtdetr2/RT-DETRv2/rtdetrv2_pytorch')
from test_training_mode import load_training_mode_model
import torch
import torchvision.transforms as T

def create_sample_predictions():
    """Create sample predictions using training mode for testing"""
    print("Loading model in training mode...")
    model, postprocessor, device = load_training_mode_model()

    # Load test image
    image_url = "http://k1.local/val2017/000000262487.webp"  # Baseball image
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert('RGB')
    w, h = image.size
    print(f"Image size: {w}x{h}")

    # Transform image
    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    im_data = transforms(image)[None].to(device)
    orig_target_sizes = torch.tensor([[w, h]], device=device)

    print("Running inference...")
    with torch.no_grad():
        outputs = model(im_data)
        results = postprocessor(outputs, orig_target_sizes)

        result = results[0]
        boxes = result['boxes'].cpu().numpy()
        scores = result['scores'].cpu().numpy()
        labels = result['labels'].cpu().numpy()

        # Create COCO format predictions
        predictions = []
        for i in range(len(boxes)):
            # Convert from [x1,y1,x2,y2] to [x,y,width,height]
            x1, y1, x2, y2 = boxes[i]
            predictions.append({
                'image_id': 262487,  # COCO image ID for this image
                'category_id': int(labels[i]),
                'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                'score': float(scores[i])
            })

        return predictions

def create_sample_ground_truth():
    """Create minimal ground truth for the test image"""
    # This is the actual ground truth for image 262487 (baseball image)
    gt_dataset = {
        'images': [
            {
                'id': 262487,
                'width': 640,
                'height': 480
            }
        ],
        'annotations': [
            {'id': 1, 'image_id': 262487, 'category_id': 1, 'bbox': [135, 111, 111, 254], 'area': 28194, 'iscrowd': 0},  # person
            {'id': 2, 'image_id': 262487, 'category_id': 1, 'bbox': [201, 276, 217, 182], 'area': 39494, 'iscrowd': 0},  # person
            {'id': 3, 'image_id': 262487, 'category_id': 1, 'bbox': [290, 276, 127, 181], 'area': 22987, 'iscrowd': 0},  # person
            {'id': 4, 'image_id': 262487, 'category_id': 39, 'bbox': [223, 93, 42, 96], 'area': 4032, 'iscrowd': 0},   # baseball_bat
        ],
        'categories': [
            {'id': 1, 'name': 'person', 'supercategory': 'person'},
            {'id': 39, 'name': 'baseball_bat', 'supercategory': 'sports'}
        ]
    }
    return gt_dataset

def test_coco_params():
    """Test different COCO evaluation parameters"""

    print("Creating sample predictions...")
    predictions = create_sample_predictions()
    print(f"Generated {len(predictions)} predictions")

    print("Creating ground truth...")
    gt_dataset = create_sample_ground_truth()

    # Save ground truth to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as gt_file:
        json.dump(gt_dataset, gt_file)
        gt_file_path = gt_file.name

    try:
        # Load ground truth
        coco_gt = COCO(gt_file_path)
        coco_dt = coco_gt.loadRes(predictions)

        print("\n" + "="*60)
        print("TESTING DEFAULT COCO PARAMETERS")
        print("="*60)

        # Test 1: Default parameters
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.imgIds = [262487]
        print(f"Default maxDets: {coco_eval.params.maxDets}")
        print(f"Default iouThrs: {coco_eval.params.iouThrs}")
        print(f"Default recThrs: {len(coco_eval.params.recThrs)} thresholds")
        print(f"Default areaRng: {coco_eval.params.areaRng}")
        print(f"Default areaRngLbl: {coco_eval.params.areaRngLbl}")

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        default_stats = coco_eval.stats.copy() if hasattr(coco_eval, 'stats') else [0.0] * 12
        print(f"Default mAP: {default_stats[0]:.3f}")

        print("\n" + "="*60)
        print("TESTING OFFICIAL RT-DETR PARAMETERS")
        print("="*60)

        # Test 2: Try parameters that might match official evaluation
        coco_eval2 = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval2.params.imgIds = [262487]

        # Try maxDets=300 to match RT-DETR output
        coco_eval2.params.maxDets = [1, 10, 300]  # Instead of [1, 10, 100]
        print(f"Modified maxDets: {coco_eval2.params.maxDets}")

        coco_eval2.evaluate()
        coco_eval2.accumulate()
        coco_eval2.summarize()

        modified_stats = coco_eval2.stats.copy() if hasattr(coco_eval2, 'stats') else [0.0] * 12
        print(f"Modified mAP: {modified_stats[0]:.3f}")

        print("\n" + "="*60)
        print("COMPARISON")
        print("="*60)
        print(f"Default mAP:  {default_stats[0]:.3f}")
        print(f"Modified mAP: {modified_stats[0]:.3f}")
        print(f"Difference:   {modified_stats[0] - default_stats[0]:.3f}")

        if abs(modified_stats[0] - default_stats[0]) > 0.001:
            print("*** PARAMETER DIFFERENCE DETECTED! ***")
        else:
            print("No significant difference from parameter changes")

    finally:
        os.unlink(gt_file_path)

if __name__ == "__main__":
    test_coco_params()