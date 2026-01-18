#!/usr/bin/env python3
"""
Check VGG16 weights from npz file
"""
import numpy as np
import sys

npz_path = sys.argv[1] if len(sys.argv) > 1 else "/home/user/.deepczero/weights/vgg16.npz"

print(f"=== Checking NPZ file: {npz_path} ===\n")

npz = np.load(npz_path)

print("Keys in npz file:")
for key in sorted(npz.keys())[:10]:
    print(f"  {key}")
print(f"  ... (showing first 10 of {len(npz.keys())} keys)\n")

# Check conv1_1 weights
if 'conv1_1/W' in npz:
    conv1_1_w = npz['conv1_1/W']
    print(f"conv1_1/W:")
    print(f"  Shape: {conv1_1_w.shape}")
    print(f"  Dtype: {conv1_1_w.dtype}")
    print(f"  Mean: {conv1_1_w.mean():.6f}")
    print(f"  First 5 values (flattened): {conv1_1_w.flatten()[:5]}")
    print(f"  Memory order: {'C-contiguous' if conv1_1_w.flags['C_CONTIGUOUS'] else 'F-contiguous' if conv1_1_w.flags['F_CONTIGUOUS'] else 'Neither'}")
    print()

if 'conv1_1/b' in npz:
    conv1_1_b = npz['conv1_1/b']
    print(f"conv1_1/b:")
    print(f"  Shape: {conv1_1_b.shape}")
    print(f"  Mean: {conv1_1_b.mean():.6f}")
    print(f"  First 5 values: {conv1_1_b[:5]}")
    print()

# Check fc6 weights
if 'fc6/W' in npz:
    fc6_w = npz['fc6/W']
    print(f"fc6/W:")
    print(f"  Shape: {fc6_w.shape}")
    print(f"  Dtype: {fc6_w.dtype}")
    print(f"  Mean: {fc6_w.mean():.6f}")
    print(f"  First 5 values (flattened): {fc6_w.flatten()[:5]}")
    print(f"  Memory order: {'C-contiguous' if fc6_w.flags['C_CONTIGUOUS'] else 'F-contiguous' if fc6_w.flags['F_CONTIGUOUS'] else 'Neither'}")
