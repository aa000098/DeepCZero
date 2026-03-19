#!/usr/bin/env python3
"""Compare C++ detection results with PyTorch reference."""
import sys
import os
import numpy as np
import torch
import glob

def main():
    hub_dir = torch.hub.get_dir()
    yolov5_dir = None
    for d in glob.glob(f"{hub_dir}/ultralytics_yolov5_*"):
        if os.path.isdir(d):
            yolov5_dir = d
            break
    sys.path.insert(0, yolov5_dir)

    # Load unfused model
    pt_path = "yolov5s.pt"
    if not os.path.exists(pt_path):
        pt_path = os.path.expanduser("~/yolov5s.pt")
    ckpt = torch.load(pt_path, map_location='cpu', weights_only=False)
    model = ckpt['model'].float().eval()

    # Load same input
    inp_npz = np.load("/tmp/yolov5_input.npz")
    inp = torch.from_numpy(inp_npz["input"])  # [1, 3, 640, 640]

    # Run unfused model to get raw outputs
    with torch.no_grad():
        # Get P3, P4, P5 from the detect head
        # model.model[0:10] = backbone, model.model[10:24] = neck, model.model[24] = detect
        x = inp
        # Full forward through backbone + neck
        y = []
        for i, m in enumerate(model.model):
            if m.f != -1:  # if from != -1, need to combine inputs
                if isinstance(m.f, int):
                    x = y[m.f]
                else:
                    x = [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x)

        # Last layer is Detect, which returns combined output
        # Let's get the raw conv outputs instead
        detect = model.model[-1]
        print(f"Detect layer: {type(detect)}")
        print(f"Detect inputs from layers: {detect.f}")

        # Get the feature maps that go into detect
        feature_maps = [y[j] for j in detect.f]
        print(f"\nFeature map shapes:")
        for i, fm in enumerate(feature_maps):
            print(f"  P{i+3}: {fm.shape}")

        # Apply detect head convolutions
        raw_outputs = []
        for i, (conv, fm) in enumerate(zip(detect.m, feature_maps)):
            out = conv(fm)
            raw_outputs.append(out)
            print(f"\nRaw P{i+3} output shape: {out.shape}")
            print(f"  P{i+3} [0,0:5,0,0]: {out[0,:5,0,0].tolist()}")

    # Now do decode + NMS using ultralytics code
    print("\n=== PyTorch NMS results (using torchvision) ===")
    try:
        from utils.general import non_max_suppression
        # Run the full model to get the combined predictions
        with torch.no_grad():
            pred = model(inp)
        # pred shape: [1, 25200, 85]
        results = non_max_suppression(pred, conf_thres=0.45, iou_thres=0.45)
        for det in results:
            if len(det):
                # Rescale to original image size
                # det format: [x1, y1, x2, y2, conf, cls]
                # These are in 640x640 space, need to rescale
                # Our letterbox: scale=0.8333, pad_left=0, pad_top=80
                det[:, 0] = (det[:, 0] - 0) / 0.833333  # x1
                det[:, 1] = (det[:, 1] - 80) / 0.833333  # y1
                det[:, 2] = (det[:, 2] - 0) / 0.833333  # x2
                det[:, 3] = (det[:, 3] - 80) / 0.833333  # y2

                # Clamp
                det[:, 0].clamp_(0, 768)
                det[:, 1].clamp_(0, 576)
                det[:, 2].clamp_(0, 768)
                det[:, 3].clamp_(0, 576)

                coco_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
                    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                    'hair drier', 'toothbrush']

                print(f"\nPyTorch detections ({len(det)}):")
                for i, d in enumerate(det):
                    cls_id = int(d[5])
                    cls_name = coco_names[cls_id] if cls_id < len(coco_names) else "unknown"
                    print(f"  [{i}] {cls_name} (conf: {d[4]:.6f}) bbox: [{d[0]:.3f}, {d[1]:.3f}, {d[2]:.3f}, {d[3]:.3f}]")
    except Exception as e:
        print(f"Could not run NMS: {e}")

    # Also compare raw head outputs
    print("\n=== Raw head comparison (unfused model) ===")
    print("C++ results (from last run):")
    print("  dog  (0.919) bbox: [134.4, 218.1, 310.2, 544.6]")
    print("  car  (0.696) bbox: [470.4, 75.4, 688.7, 173.3]")
    print("  bicycle (0.572) bbox: [159.2, 118.8, 568.6, 416.1]")
    print("  truck (0.545) bbox: [471.4, 77.6, 687.0, 171.6]")


if __name__ == "__main__":
    main()
