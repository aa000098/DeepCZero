#!/usr/bin/env python3
import numpy as np
from PIL import Image

# Preprocessing
img = Image.open("/home/user/.deepczero/data/zebra.jpg")
img = img.resize((224, 224), Image.BILINEAR)
x = np.array(img, dtype=np.float32)
x = x[:, :, ::-1]  # RGB to BGR
x[:, :, 0] -= 103.939
x[:, :, 1] -= 116.779
x[:, :, 2] -= 123.68
x = x.transpose((2, 0, 1)).reshape((1, 3, 224, 224))

print(f"Input: shape={x.shape}, range=[{x.min():.3f}, {x.max():.3f}]")

# Load weights
npz = np.load("/home/user/.deepczero/weights/vgg16.npz")

def relu(x):
    return np.maximum(0, x)

def conv2d_scipy(x, W, b):
    """Conv2d using scipy.signal.correlate2d"""
    from scipy.signal import correlate2d
    N, C_in, H, W_dim = x.shape
    C_out, C_in2, KH, KW = W.shape

    out = np.zeros((N, C_out, H, W_dim), dtype=np.float32)

    for n in range(N):
        for c_out in range(C_out):
            for c_in in range(C_in):
                # 'same' mode with boundary='fill' (zero padding)
                corr = correlate2d(x[n, c_in], W[c_out, c_in], mode='same', boundary='fill', fillvalue=0)
                out[n, c_out] += corr
            out[n, c_out] += b[c_out]

    return out

def pool2d(x):
    """Max pool 2x2 stride 2"""
    N, C, H, W = x.shape
    out_H = H // 2
    out_W = W // 2
    out = np.zeros((N, C, out_H, out_W), dtype=np.float32)

    for n in range(N):
        for c in range(C):
            for h in range(out_H):
                for w in range(out_W):
                    out[n, c, h, w] = np.max(x[n, c, h*2:h*2+2, w*2:w*2+2])

    return out

print("\n=== Running VGG16 forward (NumPy + scipy) ===\n")

# Block 1
x = conv2d_scipy(x, npz['conv1_1/W'], npz['conv1_1/b'])
x = relu(x)
print(f"After conv1_1: {x.shape}, range=[{x.min():.1f}, {x.max():.1f}]")

x = conv2d_scipy(x, npz['conv1_2/W'], npz['conv1_2/b'])
x = relu(x)
x = pool2d(x)
print(f"After pool1: {x.shape}, range=[{x.min():.1f}, {x.max():.1f}]")

# Block 2
x = conv2d_scipy(x, npz['conv2_1/W'], npz['conv2_1/b'])
x = relu(x)
x = conv2d_scipy(x, npz['conv2_2/W'], npz['conv2_2/b'])
x = relu(x)
x = pool2d(x)
print(f"After pool2: {x.shape}, range=[{x.min():.1f}, {x.max():.1f}]")

# Block 3
x = conv2d_scipy(x, npz['conv3_1/W'], npz['conv3_1/b'])
x = relu(x)
x = conv2d_scipy(x, npz['conv3_2/W'], npz['conv3_2/b'])
x = relu(x)
x = conv2d_scipy(x, npz['conv3_3/W'], npz['conv3_3/b'])
x = relu(x)
x = pool2d(x)
print(f"After pool3: {x.shape}, range=[{x.min():.1f}, {x.max():.1f}]")

# Block 4
x = conv2d_scipy(x, npz['conv4_1/W'], npz['conv4_1/b'])
x = relu(x)
x = conv2d_scipy(x, npz['conv4_2/W'], npz['conv4_2/b'])
x = relu(x)
x = conv2d_scipy(x, npz['conv4_3/W'], npz['conv4_3/b'])
x = relu(x)
x = pool2d(x)
print(f"After pool4: {x.shape}, range=[{x.min():.1f}, {x.max():.1f}]")

# Block 5
x = conv2d_scipy(x, npz['conv5_1/W'], npz['conv5_1/b'])
x = relu(x)
x = conv2d_scipy(x, npz['conv5_2/W'], npz['conv5_2/b'])
x = relu(x)
x = conv2d_scipy(x, npz['conv5_3/W'], npz['conv5_3/b'])
x = relu(x)
x = pool2d(x)
print(f"After pool5: {x.shape}, range=[{x.min():.1f}, {x.max():.1f}]")

# Flatten
x = x.reshape(1, -1)
print(f"After flatten: {x.shape}")

# FC6
fc6_W = np.ascontiguousarray(npz['fc6/W'])
x = np.dot(x, fc6_W) + npz['fc6/b']
x = relu(x)
print(f"After fc6: {x.shape}, range=[{x.min():.1f}, {x.max():.1f}]")

# FC7
fc7_W = np.ascontiguousarray(npz['fc7/W'])
x = np.dot(x, fc7_W) + npz['fc7/b']
x = relu(x)
print(f"After fc7: {x.shape}, range=[{x.min():.1f}, {x.max():.1f}]")

# FC8
fc8_W = np.ascontiguousarray(npz['fc8/W'])
x = np.dot(x, fc8_W) + npz['fc8/b']
print(f"After fc8: {x.shape}, range=[{x.min():.1f}, {x.max():.1f}]")

# Top-5
top5 = np.argsort(x[0])[::-1][:5]

print("\n=== Top-5 Predictions (Python) ===")
for i, idx in enumerate(top5, 1):
    print(f"  {i}. Class {idx}: {x[0, idx]:.4f}")
