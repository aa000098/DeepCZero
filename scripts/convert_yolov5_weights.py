#!/usr/bin/env python3
"""
Convert ultralytics YOLOv5s weights (.pt) to numpy (.npz) format
for DeepCZero framework.

Usage:
    pip install torch
    python scripts/convert_yolov5_weights.py [yolov5s.pt]

Requires yolov5s.pt in current directory (or specify path).
Downloads YOLOv5 source via torch.hub for model class definitions.

Output: ~/.deepczero/weights/yolov5s.npz
"""

import sys
import os
import numpy as np


def convert_yolov5s(pt_path="yolov5s.pt", output_path=None):
    try:
        import torch
    except ImportError:
        print("Error: PyTorch is required. Install with: pip install torch")
        sys.exit(1)

    if not os.path.exists(pt_path):
        print(f"Error: {pt_path} not found.")
        print("Download from: https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt")
        sys.exit(1)

    # Default output path
    if output_path is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".deepczero", "weights")
        os.makedirs(cache_dir, exist_ok=True)
        output_path = os.path.join(cache_dir, "yolov5s.npz")

    # Ensure YOLOv5 hub source is available (needed to unpickle the .pt)
    print("Ensuring YOLOv5 source is available...")
    hub_dir = torch.hub.get_dir()
    yolov5_dir = None
    import glob
    for d in glob.glob(f"{hub_dir}/ultralytics_yolov5_*"):
        if os.path.isdir(d):
            yolov5_dir = d
            break

    if yolov5_dir is None:
        print("Downloading YOLOv5 source via torch.hub...")
        torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=False, _verbose=False)
        for d in glob.glob(f"{hub_dir}/ultralytics_yolov5_*"):
            if os.path.isdir(d):
                yolov5_dir = d
                break

    if yolov5_dir is None:
        print("Error: Could not find YOLOv5 source directory")
        sys.exit(1)

    print(f"Using YOLOv5 source: {yolov5_dir}")
    sys.path.insert(0, yolov5_dir)

    # Load raw (unfused) checkpoint
    print(f"Loading raw checkpoint: {pt_path}")
    ckpt = torch.load(pt_path, map_location='cpu', weights_only=False)
    model = ckpt['model'].float()
    sd = model.state_dict()

    print(f"Raw state_dict: {len(sd)} keys")

    out = {}
    skipped = 0
    for key, tensor in sd.items():
        arr = tensor.cpu().float().numpy()
        npz_key = key

        # Skip num_batches_tracked (not needed for inference)
        if 'num_batches_tracked' in key:
            skipped += 1
            continue

        # Skip anchor buffers (defined in C++ postprocess code)
        if '.anchors' in key or '.anchor_grid' in key:
            skipped += 1
            continue

        # Conv2d weight: .conv.weight -> .conv.W
        if '.conv.weight' in key:
            npz_key = key.replace('.conv.weight', '.conv.W')
        # Detection head Conv2d: model.24.m.X.weight -> model.24.m.X.W
        elif key.startswith('model.24.m.') and key.endswith('.weight') and '.bn.' not in key:
            npz_key = key.replace('.weight', '.W')
        elif key.startswith('model.24.m.') and key.endswith('.bias') and '.bn.' not in key:
            npz_key = key.replace('.bias', '.b')
        # BN keys (weight, bias, running_mean, running_var) stay unchanged

        out[npz_key] = arr
        print(f"  {key:55s} -> {npz_key:55s}  shape={list(arr.shape)}")

    np.savez(output_path, **out)
    print(f"\nSaved {len(out)} tensors (skipped {skipped}) to {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")

    # Verify
    print("\nVerification:")
    loaded = np.load(output_path)
    print(f"  Keys in npz: {len(loaded.files)}")

    expected = [
        "model.0.conv.W", "model.0.bn.weight", "model.0.bn.bias",
        "model.0.bn.running_mean", "model.0.bn.running_var",
        "model.24.m.0.W", "model.24.m.0.b",
    ]
    for k in expected:
        if k in loaded:
            print(f"  OK: {k} shape={loaded[k].shape}")
        else:
            print(f"  MISSING: {k}")


if __name__ == "__main__":
    pt = sys.argv[1] if len(sys.argv) > 1 else "yolov5s.pt"
    out = sys.argv[2] if len(sys.argv) > 2 else None
    convert_yolov5s(pt, out)
