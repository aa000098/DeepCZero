#!/usr/bin/env python3
"""
Debug script: save detailed intermediate outputs for backbone_0 (Conv->BN->SiLU).
Compare these with C++ to find where values diverge.
"""
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import glob

def main():
    # Load YOLOv5 source (needed to unpickle .pt)
    hub_dir = torch.hub.get_dir()
    yolov5_dir = None
    for d in glob.glob(f"{hub_dir}/ultralytics_yolov5_*"):
        if os.path.isdir(d):
            yolov5_dir = d
            break
    if yolov5_dir is None:
        print("Error: YOLOv5 source not found. Run convert_yolov5_weights.py first.")
        sys.exit(1)
    sys.path.insert(0, yolov5_dir)

    # Load raw unfused model
    pt_path = "yolov5s.pt"
    if not os.path.exists(pt_path):
        pt_path = os.path.expanduser("~/yolov5s.pt")
    print(f"Loading checkpoint: {pt_path}")
    ckpt = torch.load(pt_path, map_location='cpu', weights_only=False)
    model = ckpt['model'].float().eval()

    # Load the same preprocessed input
    inp_npz = np.load("/tmp/yolov5_input.npz")
    inp = torch.from_numpy(inp_npz["input"])  # [1, 3, 480, 640]
    print(f"Input shape: {inp.shape}")
    print(f"Input [0,0,0,0:5]: {inp[0,0,0,:5].tolist()}")

    # backbone_0 = model.model[0] which is a CBS (Conv-BN-SiLU)
    layer0 = model.model[0]
    print(f"\nLayer 0 type: {type(layer0)}")
    print(f"  conv: {layer0.conv}")
    print(f"  bn:   {type(layer0.bn) if hasattr(layer0, 'bn') else 'N/A'}")
    print(f"  act:  {type(layer0.act) if hasattr(layer0, 'act') else 'N/A'}")

    with torch.no_grad():
        # Step 1: Conv only
        conv_out = layer0.conv(inp)
        print(f"\nConv output shape: {conv_out.shape}")
        print(f"Conv [0,0:5,0,0]: {conv_out[0,:5,0,0].tolist()}")
        print(f"Conv [0,0:5,120,160]: {conv_out[0,:5,120,160].tolist()}")

        # Step 2: BN
        bn_out = layer0.bn(conv_out)
        print(f"\nBN output shape: {bn_out.shape}")
        print(f"BN [0,0:5,0,0]: {bn_out[0,:5,0,0].tolist()}")
        print(f"BN [0,0:5,120,160]: {bn_out[0,:5,120,160].tolist()}")

        # Step 3: SiLU
        silu_out = layer0.act(bn_out)
        print(f"\nSiLU output shape: {silu_out.shape}")
        print(f"SiLU [0,0:5,0,0]: {silu_out[0,:5,0,0].tolist()}")
        print(f"SiLU [0,0:5,120,160]: {silu_out[0,:5,120,160].tolist()}")

        # Full layer0 output (should equal SiLU output)
        full_out = layer0(inp)
        print(f"\nFull layer0 output shape: {full_out.shape}")
        print(f"Full [0,0:5,0,0]: {full_out[0,:5,0,0].tolist()}")

        # Print weight info for verification
        conv_w = layer0.conv.weight.data
        print(f"\n=== Weight verification ===")
        print(f"Conv W shape: {conv_w.shape}")
        print(f"Conv W [0,0,0,0:6]: {conv_w[0,0,0,:].tolist()}")
        print(f"Conv W [0,0,0:6,0]: {conv_w[0,0,:,0].tolist()}")
        print(f"Conv W first 5 flat: {conv_w.flatten()[:5].tolist()}")
        print(f"Conv has bias: {layer0.conv.bias is not None}")

        if hasattr(layer0, 'bn'):
            bn = layer0.bn
            print(f"\nBN weight first 5: {bn.weight.data[:5].tolist()}")
            print(f"BN bias first 5: {bn.bias.data[:5].tolist()}")
            print(f"BN running_mean first 5: {bn.running_mean[:5].tolist()}")
            print(f"BN running_var first 5: {bn.running_var[:5].tolist()}")
            print(f"BN eps: {bn.eps}")

    # Save intermediates
    np.savez("/tmp/yolov5_debug_layer0.npz",
             conv_out=conv_out.numpy(),
             bn_out=bn_out.numpy(),
             silu_out=silu_out.numpy(),
             conv_w=conv_w.numpy(),
             bn_weight=layer0.bn.weight.data.numpy() if hasattr(layer0, 'bn') else np.array([]),
             bn_bias=layer0.bn.bias.data.numpy() if hasattr(layer0, 'bn') else np.array([]),
             bn_running_mean=layer0.bn.running_mean.numpy() if hasattr(layer0, 'bn') else np.array([]),
             bn_running_var=layer0.bn.running_var.numpy() if hasattr(layer0, 'bn') else np.array([]))
    print("\nSaved to /tmp/yolov5_debug_layer0.npz")


if __name__ == "__main__":
    main()
