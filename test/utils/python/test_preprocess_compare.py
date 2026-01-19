#!/usr/bin/env python3
"""
VGG16 preprocessing comparison test
"""
import numpy as np
from PIL import Image

def preprocess_vgg16(image_path):
    """VGG16 preprocessing (same as DeZero)"""
    # 1. Load image
    img = Image.open(image_path)
    print(f"Original image size: {img.size}")

    # 2. Resize to 224x224 (using different methods)
    print(f"\nTrying different resize methods:")

    # LANCZOS (PIL default for downsampling)
    img_lanczos = img.resize((224, 224), Image.LANCZOS)
    img_array_lanczos = np.array(img_lanczos, dtype=np.float32)

    # BILINEAR (similar to stb linear)
    img_bilinear = img.resize((224, 224), Image.BILINEAR)
    img_array = np.array(img_bilinear, dtype=np.float32)

    print(f"  Using BILINEAR (similar to stb)")
    print(f"  Resized shape: {img_array.shape}")
    print(f"  Center pixel before processing: R={img_array[112, 112, 0]:.2f}, G={img_array[112, 112, 1]:.2f}, B={img_array[112, 112, 2]:.2f}")

    # 3. RGB -> BGR and mean subtraction
    img_array = img_array[:, :, ::-1]  # RGB to BGR
    img_array[:, :, 0] -= 103.939  # B
    img_array[:, :, 1] -= 116.779  # G
    img_array[:, :, 2] -= 123.68   # R

    # 4. (H, W, C) -> (C, H, W)
    img_array = img_array.transpose((2, 0, 1))

    # 5. Add batch dimension: (C, H, W) -> (1, C, H, W)
    img_array = img_array.reshape((1, 3, 224, 224))

    return img_array

def main():
    print("=== VGG16 Preprocessing Test (Python) ===\n")

    img_path = "/home/user/.deepczero/data/zebra.jpg"

    # Preprocess
    x = preprocess_vgg16(img_path)

    print(f"\n=== Preprocessed Result ===")
    print(f"Shape: {x.shape}")
    print(f"Min: {x.min():.6f}, Max: {x.max():.6f}, Mean: {x.mean():.6f}")

    print(f"\n=== Sample pixels [0, :, 0, 0] ===")
    print(f"B channel: {x[0, 0, 0, 0]:.6f}")
    print(f"G channel: {x[0, 1, 0, 0]:.6f}")
    print(f"R channel: {x[0, 2, 0, 0]:.6f}")

    print(f"\n=== First 5 pixels of B channel [0, 0, 0, :5] ===")
    print(x[0, 0, 0, :5])

    print(f"\n=== Center pixel [0, :, 112, 112] ===")
    print(f"B: {x[0, 0, 112, 112]:.6f}")
    print(f"G: {x[0, 1, 112, 112]:.6f}")
    print(f"R: {x[0, 2, 112, 112]:.6f}")

if __name__ == "__main__":
    main()
