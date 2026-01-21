#!/usr/bin/env python3
"""Test F-contiguous to C-contiguous conversion"""
import numpy as np

# Load the weights
npz = np.load("/home/user/.deepczero/weights/vgg16.npz")
fc6_w = npz['fc6/W']

print(f"Original array:")
print(f"  Shape: {fc6_w.shape}")
print(f"  Fortran order: {fc6_w.flags['F_CONTIGUOUS']}")
print(f"  C order: {fc6_w.flags['C_CONTIGUOUS']}")
print(f"  First 10 values (arr.flatten('C')): {fc6_w.flatten('C')[:10]}")
print(f"  First 10 values (arr.flatten('F')): {fc6_w.flatten('F')[:10]}")

print(f"\n  arr[0,0] = {fc6_w[0,0]}")
print(f"  arr[1,0] = {fc6_w[1,0]}")
print(f"  arr[0,1] = {fc6_w[0,1]}")

# Method 1: Just convert to C-contiguous
fc6_w_c = np.ascontiguousarray(fc6_w)
print(f"\nMethod 1: np.ascontiguousarray()")
print(f"  Shape: {fc6_w_c.shape}")
print(f"  C order: {fc6_w_c.flags['C_CONTIGUOUS']}")
print(f"  First 10 values: {fc6_w_c.flatten('C')[:10]}")
print(f"  arr[0,0] = {fc6_w_c[0,0]}")
print(f"  arr[1,0] = {fc6_w_c[1,0]}")
print(f"  arr[0,1] = {fc6_w_c[0,1]}")

# Method 2: Read raw bytes as if F-order, reshape to transposed shape, then transpose back
raw_bytes = fc6_w.tobytes('A')  # Raw bytes in actual memory order (F-order)
print(f"\nMethod 2: Read raw F-order bytes, reshape as transposed, then transpose")
# If we interpret F-order (25088, 4096) memory as C-order, it becomes (4096, 25088)
fc6_w_transposed = np.frombuffer(raw_bytes, dtype=np.float32).reshape(4096, 25088)
print(f"  Shape after reshape: {fc6_w_transposed.shape}")
print(f"  First 10 values: {fc6_w_transposed.flatten('C')[:10]}")
fc6_w_fixed = fc6_w_transposed.T
print(f"  Shape after transpose: {fc6_w_fixed.shape}")
print(f"  First 10 values: {fc6_w_fixed.flatten('C')[:10]}")
print(f"  arr[0,0] = {fc6_w_fixed[0,0]}")
print(f"  arr[1,0] = {fc6_w_fixed[1,0]}")
print(f"  arr[0,1] = {fc6_w_fixed[0,1]}")

print(f"\nAre they equal? {np.allclose(fc6_w_c, fc6_w_fixed)}")
