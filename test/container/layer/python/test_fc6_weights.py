#!/usr/bin/env python3
import numpy as np

npz = np.load("/home/user/.deepczero/weights/vgg16.npz")
fc6_w = npz['fc6/W']

print(f"fc6/W shape: {fc6_w.shape}")
print(f"Memory order: {'C' if fc6_w.flags['C_CONTIGUOUS'] else 'F' if fc6_w.flags['F_CONTIGUOUS'] else '?'}")
print(f"\nFirst 10 values when read as C-contiguous (row-major):")
print(fc6_w.flatten('C')[:10])
print(f"\nFirst 10 values when read as F-contiguous (column-major):")
print(fc6_w.flatten('F')[:10])
print(f"\nfc6_w[0, :5] (first row, first 5 cols):")
print(fc6_w[0, :5])
print(f"\nfc6_w[:5, 0] (first col, first 5 rows):")
print(fc6_w[:5, 0])
