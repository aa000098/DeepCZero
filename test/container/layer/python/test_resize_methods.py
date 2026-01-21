#!/usr/bin/env python3
"""
Test different resize methods
"""
import numpy as np
from PIL import Image

img_path = "/home/user/.deepczero/data/zebra.jpg"
img = Image.open(img_path)

print(f"Original size: {img.size}")
print(f"\nTesting center pixel (112, 112) with different resize methods:\n")

methods = {
    "NEAREST": Image.NEAREST,
    "BILINEAR": Image.BILINEAR,
    "BICUBIC": Image.BICUBIC,
    "LANCZOS": Image.LANCZOS,
}

mean_b, mean_g, mean_r = 103.939, 116.779, 123.68

for name, method in methods.items():
    resized = img.resize((224, 224), method)
    arr = np.array(resized, dtype=np.float32)

    # BGR order and mean subtraction
    r, g, b = arr[112, 112, :]
    b_processed = b - mean_b
    g_processed = g - mean_g
    r_processed = r - mean_r

    print(f"{name:10} - Center pixel: B={b_processed:7.3f}, G={g_processed:7.3f}, R={r_processed:7.3f}")

print(f"\nPIL default (when no method specified) is usually BILINEAR for downsampling")
print(f"But older versions used NEAREST, newer use LANCZOS")

# Test with default
resized_default = img.resize((224, 224))
arr_default = np.array(resized_default, dtype=np.float32)
r, g, b = arr_default[112, 112, :]
print(f"\nDefault resize: B={b-mean_b:7.3f}, G={g-mean_g:7.3f}, R={r-mean_r:7.3f}")
