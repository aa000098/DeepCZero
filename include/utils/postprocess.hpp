#pragma once

#include "container/variable.hpp"
#include "utils/preprocess.hpp"

#include <vector>
#include <string>

struct Detection {
    float x1, y1, x2, y2;
    float confidence;
    int class_id;
};

// YOLOv5 anchors
static const std::vector<std::pair<float, float>> YOLOV5_ANCHORS_P3 = {{10,13}, {16,30}, {33,23}};
static const std::vector<std::pair<float, float>> YOLOV5_ANCHORS_P4 = {{30,61}, {62,45}, {59,119}};
static const std::vector<std::pair<float, float>> YOLOV5_ANCHORS_P5 = {{116,90}, {156,198}, {373,326}};

// Decode all 3 detection heads
std::vector<Detection> decode_yolov5_outputs(
    const std::vector<Variable>& outputs,
    size_t num_classes = 80,
    float conf_threshold = 0.25f);

// Non-Maximum Suppression
std::vector<Detection> nms(
    std::vector<Detection>& detections,
    float iou_threshold = 0.45f);

// Rescale from letterbox to original image coordinates
void rescale_detections(
    std::vector<Detection>& detections,
    const LetterboxInfo& info);

// Draw bounding boxes on image and save
void draw_detections(
    const std::string& input_image_path,
    const std::vector<Detection>& detections,
    const std::string& output_path);

// COCO 80 class names
static const std::vector<std::string> COCO_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};
