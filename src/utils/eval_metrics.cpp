#include "utils/eval_metrics.hpp"

#include <algorithm>
#include <numeric>
#include <cmath>


float compute_iou_box(float x1a, float y1a, float x2a, float y2a,
					  float x1b, float y1b, float x2b, float y2b) {
	float inter_x1 = std::max(x1a, x1b);
	float inter_y1 = std::max(y1a, y1b);
	float inter_x2 = std::min(x2a, x2b);
	float inter_y2 = std::min(y2a, y2b);

	float inter_w = std::max(0.0f, inter_x2 - inter_x1);
	float inter_h = std::max(0.0f, inter_y2 - inter_y1);
	float inter_area = inter_w * inter_h;

	float area_a = (x2a - x1a) * (y2a - y1a);
	float area_b = (x2b - x1b) * (y2b - y1b);
	float union_area = area_a + area_b - inter_area;

	if (union_area <= 0.0f) return 0.0f;
	return inter_area / union_area;
}


// 11-point interpolation AP
static float compute_ap_11point(const std::vector<float>& precisions,
								const std::vector<float>& recalls) {
	float ap = 0.0f;
	for (float t = 0.0f; t <= 1.0f; t += 0.1f) {
		float max_prec = 0.0f;
		for (size_t i = 0; i < recalls.size(); i++) {
			if (recalls[i] >= t) {
				max_prec = std::max(max_prec, precisions[i]);
			}
		}
		ap += max_prec;
	}
	return ap / 11.0f;
}


struct PredWithImage {
	size_t image_id;
	float confidence;
	float x1, y1, x2, y2;
};


APResult compute_map(const std::vector<std::vector<Detection>>& all_preds,
					 const std::vector<std::vector<Detection>>& all_gts,
					 size_t num_classes,
					 float iou_threshold) {
	APResult result;
	result.ap_per_class.resize(num_classes, 0.0f);

	size_t num_images = all_gts.size();
	int classes_with_gt = 0;

	for (size_t c = 0; c < num_classes; c++) {
		// Collect all predictions for this class
		std::vector<PredWithImage> class_preds;
		for (size_t img = 0; img < all_preds.size(); img++) {
			for (const auto& det : all_preds[img]) {
				if (static_cast<size_t>(det.class_id) == c) {
					class_preds.push_back({img, det.confidence,
										   det.x1, det.y1, det.x2, det.y2});
				}
			}
		}

		// Collect all GT for this class per image
		size_t total_gt = 0;
		std::vector<std::vector<Detection>> gt_per_image(num_images);
		for (size_t img = 0; img < num_images; img++) {
			for (const auto& gt : all_gts[img]) {
				if (static_cast<size_t>(gt.class_id) == c) {
					gt_per_image[img].push_back(gt);
					total_gt++;
				}
			}
		}

		if (total_gt == 0) continue;
		classes_with_gt++;

		if (class_preds.empty()) {
			result.ap_per_class[c] = 0.0f;
			continue;
		}

		// Sort by confidence descending
		std::sort(class_preds.begin(), class_preds.end(),
				  [](const PredWithImage& a, const PredWithImage& b) {
					  return a.confidence > b.confidence;
				  });

		// Track which GTs have been matched per image
		std::vector<std::vector<bool>> matched(num_images);
		for (size_t img = 0; img < num_images; img++) {
			matched[img].resize(gt_per_image[img].size(), false);
		}

		// Compute TP/FP
		std::vector<float> precisions, recalls;
		size_t tp = 0, fp = 0;

		for (const auto& pred : class_preds) {
			size_t img = pred.image_id;
			float best_iou = 0.0f;
			int best_gt = -1;

			for (size_t g = 0; g < gt_per_image[img].size(); g++) {
				if (matched[img][g]) continue;
				const auto& gt = gt_per_image[img][g];
				float iou = compute_iou_box(pred.x1, pred.y1, pred.x2, pred.y2,
											gt.x1, gt.y1, gt.x2, gt.y2);
				if (iou > best_iou) {
					best_iou = iou;
					best_gt = static_cast<int>(g);
				}
			}

			if (best_iou >= iou_threshold && best_gt >= 0) {
				tp++;
				matched[img][best_gt] = true;
			} else {
				fp++;
			}

			float precision = static_cast<float>(tp) / static_cast<float>(tp + fp);
			float recall = static_cast<float>(tp) / static_cast<float>(total_gt);
			precisions.push_back(precision);
			recalls.push_back(recall);
		}

		result.ap_per_class[c] = compute_ap_11point(precisions, recalls);
	}

	// mAP = mean over classes that have GT
	if (classes_with_gt > 0) {
		float sum = 0.0f;
		for (size_t c = 0; c < num_classes; c++) {
			// Only count classes that have GT
			bool has_gt = false;
			for (size_t img = 0; img < num_images; img++) {
				for (const auto& gt : all_gts[img]) {
					if (static_cast<size_t>(gt.class_id) == c) { has_gt = true; break; }
				}
				if (has_gt) break;
			}
			if (has_gt) sum += result.ap_per_class[c];
		}
		result.map50 = sum / static_cast<float>(classes_with_gt);
	} else {
		result.map50 = 0.0f;
	}

	return result;
}
