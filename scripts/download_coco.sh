#!/bin/bash
# Download COCO 2017 dataset for YOLOv5 training
# Usage:
#   ./scripts/download_coco.sh          # download val2017 only (small, for testing)
#   ./scripts/download_coco.sh --train   # download train2017 + val2017 (large, ~20GB)

set -e

DATA_DIR="data/coco"
mkdir -p "$DATA_DIR/images"
mkdir -p "$DATA_DIR/annotations"

echo "=== COCO 2017 Dataset Download ==="

# Always download val2017 (1GB images + 241MB annotations)
if [ ! -d "$DATA_DIR/images/val2017" ]; then
    echo "[1/2] Downloading val2017 images (~1GB)..."
    wget -q --show-progress -O "$DATA_DIR/val2017.zip" \
        "http://images.cocodataset.org/zips/val2017.zip"
    echo "Extracting..."
    unzip -q "$DATA_DIR/val2017.zip" -d "$DATA_DIR/images/"
    rm "$DATA_DIR/val2017.zip"
    echo "val2017 images: done ($(ls "$DATA_DIR/images/val2017" | wc -l) images)"
else
    echo "[1/2] val2017 images already exist, skipping."
fi

if [ ! -f "$DATA_DIR/annotations/instances_val2017.json" ]; then
    echo "[2/2] Downloading annotations (~241MB)..."
    wget -q --show-progress -O "$DATA_DIR/annotations_trainval2017.zip" \
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    echo "Extracting..."
    unzip -q -o "$DATA_DIR/annotations_trainval2017.zip" -d "$DATA_DIR/"
    rm "$DATA_DIR/annotations_trainval2017.zip"
    echo "Annotations: done"
else
    echo "[2/2] Annotations already exist, skipping."
fi

# Optionally download train2017
if [ "$1" = "--train" ]; then
    if [ ! -d "$DATA_DIR/images/train2017" ]; then
        echo "[Extra] Downloading train2017 images (~18GB)..."
        wget -q --show-progress -O "$DATA_DIR/train2017.zip" \
            "http://images.cocodataset.org/zips/train2017.zip"
        echo "Extracting..."
        unzip -q "$DATA_DIR/train2017.zip" -d "$DATA_DIR/images/"
        rm "$DATA_DIR/train2017.zip"
        echo "train2017 images: done ($(ls "$DATA_DIR/images/train2017" | wc -l) images)"
    else
        echo "[Extra] train2017 images already exist, skipping."
    fi
fi

echo ""
echo "=== Download Complete ==="
echo "Directory structure:"
echo "  $DATA_DIR/"
echo "  ├── images/"
ls -d "$DATA_DIR/images"/*/ 2>/dev/null | while read d; do
    name=$(basename "$d")
    count=$(ls "$d" | wc -l)
    echo "  │   └── $name/ ($count images)"
done
echo "  └── annotations/"
ls "$DATA_DIR/annotations/"*.json 2>/dev/null | while read f; do
    name=$(basename "$f")
    size=$(du -h "$f" | cut -f1)
    echo "      └── $name ($size)"
done
