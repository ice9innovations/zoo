#!/usr/bin/env python3
"""
Quick test script to verify RT-DETRv2 setup
"""

import sys
import os
import torch

# Add RT-DETRv2 to path
rtdetrv2_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RT-DETRv2')
sys.path.insert(0, rtdetrv2_path)

def test_pytorch():
    """Test PyTorch installation"""
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ CUDA device: {torch.cuda.get_device_name(0)}")
    return True

def test_rtdetrv2_import():
    """Test RT-DETRv2 imports"""
    try:
        from rtdetrv2_pytorch.src.core import YAMLConfig
        print("✓ RT-DETRv2 core imports successful")
        return True
    except ImportError as e:
        print(f"✗ RT-DETRv2 import failed: {e}")
        return False

def test_torch_hub():
    """Test torch.hub model loading"""
    try:
        # Try loading the smallest model for testing
        print("📦 Loading RT-DETRv2 model via torch.hub...")
        model = torch.hub.load(rtdetrv2_path, 'rtdetrv2_s', source='local', pretrained=True)
        print("✓ RT-DETRv2 model loaded successfully")

        # Test inference
        print("🧪 Testing inference...")
        dummy_input = torch.randn(1, 3, 640, 640)
        dummy_size = torch.tensor([[640, 640]])

        with torch.no_grad():
            outputs = model(dummy_input, dummy_size)
            labels, boxes, scores = outputs
            print(f"✓ Inference successful - found {len(labels)} detections")

        return True
    except Exception as e:
        print(f"✗ torch.hub model loading failed: {e}")
        return False

def main():
    print("🔍 Testing RT-DETRv2 setup...")
    print()

    tests = [
        ("PyTorch", test_pytorch),
        ("RT-DETRv2 Imports", test_rtdetrv2_import),
        ("Model Loading", test_torch_hub),
    ]

    results = []
    for name, test_func in tests:
        print(f"Testing {name}...")
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"✗ {name} failed with exception: {e}")
            results.append((name, False))
        print()

    print("📋 Test Results:")
    print("-" * 30)
    all_passed = True
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{name:<20} {status}")
        if not success:
            all_passed = False

    print()
    if all_passed:
        print("🎉 All tests passed! RT-DETRv2 is ready to use.")
        print("Next step: Create .env file and run 'python3 REST.py'")
    else:
        print("❌ Some tests failed. Check the error messages above.")
        print("Make sure you have the required dependencies installed.")

if __name__ == "__main__":
    main()