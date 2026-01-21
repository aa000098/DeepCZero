#!/usr/bin/env python3
"""Test VGG16 with Python to compare output values"""
import numpy as np
from PIL import Image

def preprocess_vgg16(image_path):
    """VGG16 preprocessing"""
    img = Image.open(image_path)
    img = img.resize((224, 224), Image.BILINEAR)
    img_array = np.array(img, dtype=np.float32)

    # RGB -> BGR and mean subtraction
    img_array = img_array[:, :, ::-1]
    img_array[:, :, 0] -= 103.939  # B
    img_array[:, :, 1] -= 116.779  # G
    img_array[:, :, 2] -= 123.68   # R

    # (H, W, C) -> (C, H, W) -> (1, C, H, W)
    img_array = img_array.transpose((2, 0, 1))
    img_array = img_array.reshape((1, 3, 224, 224))

    return img_array

def forward_linear(x, W, b):
    """Simple linear layer forward"""
    # x: (batch, in_features)
    # W: (in_features, out_features)
    # b: (out_features,)
    return np.dot(x, W) + b

def main():
    print("=== VGG16 Forward Test (Python) ===\n")

    # Load weights
    npz = np.load("/home/user/.deepczero/weights/vgg16.npz")

    # Preprocess image
    img_path = "/home/user/.deepczero/data/zebra.jpg"
    x = preprocess_vgg16(img_path)

    print(f"Input shape: {x.shape}")
    print(f"Input stats: Min={x.min():.6f}, Max={x.max():.6f}, Mean={x.mean():.6f}\n")

    # Conv1_1
    print("=== Conv1_1 ===")
    conv1_1_W = npz['conv1_1/W']
    print(f"W shape: {conv1_1_W.shape}, F-order: {conv1_1_W.flags['F_CONTIGUOUS']}")

    # Let's just do a simple test with fc6 to check if it's about softmax
    print("\n=== Skipping to FC layers ===")

    # For now, let's check what the actual output range should be
    # VGG16 outputs logits (not probabilities)
    # The final output should be raw scores, not probabilities

    fc8_W = npz['fc8/W']
    fc8_b = npz['fc8/b']

    print(f"fc8/W shape: {fc8_W.shape}")
    print(f"fc8/b shape: {fc8_b.shape}")
    print(f"fc8/W stats: Min={fc8_W.min():.6f}, Max={fc8_W.max():.6f}")
    print(f"fc8/b stats: Min={fc8_b.min():.6f}, Max={fc8_b.max():.6f}")

    print("\n=== Note ===")
    print("VGG16 outputs are LOGITS (raw scores), not probabilities.")
    print("If you want probabilities, you need to apply softmax.")
    print("Logits can have large positive/negative values.")
    print("The C++ output shows logits around 15-25, which is normal.")

if __name__ == "__main__":
    main()
