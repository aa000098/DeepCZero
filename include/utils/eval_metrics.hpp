#pragma once

#include "utils/postprocess.hpp"
#include "container/loss/yolov5_loss.hpp"

#include <vector>

struct APResult {
	float map50;
	std::vector<float> ap_per_class;
};

// IoU between two boxes in [x1,y1,x2,y2] format
float compute_iou_box(float x1a, float y1a, float x2a, float y2a,
					  float x1b, float y1b, float x2b, float y2b);

// Compute mAP@IoU from per-image predictions and ground truths
// all_preds[i]: predictions for image i (Detection with x1,y1,x2,y2,confidence,class_id)
// all_gts[i]: ground truths for image i (Detection with x1,y1,x2,y2,class_id; confidence unused)
APResult compute_map(const std::vector<std::vector<Detection>>& all_preds,
					 const std::vector<std::vector<Detection>>& all_gts,
					 size_t num_classes = 80,
					 float iou_threshold = 0.5f);
