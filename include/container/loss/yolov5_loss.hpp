#pragma once

#include "container/variable.hpp"
#include "utils/postprocess.hpp"

#include <vector>
#include <array>

// Ground truth format per image: {class_id, cx, cy, w, h} normalized to [0,1]
using GroundTruth = std::vector<std::array<float, 5>>;

struct YOLOv5Targets {
    Tensor<> target_obj;     // [B, A, H, W] binary objectness mask
    Tensor<> target_cls;     // [N_pos, num_classes] one-hot class targets
    Tensor<> target_box;     // [N_pos, 4] grid-relative box targets
    std::vector<size_t> flat_indices; // index into [B*A*H*W] for positives
};

struct YOLOv5LossResult {
    Variable total_loss;
    float box_loss_val;
    float obj_loss_val;
    float cls_loss_val;
};

class YOLOv5Loss {
private:
    size_t num_classes;
    float box_weight;
    float obj_weight;
    float cls_weight;
    float anchor_threshold;

    static constexpr size_t NUM_ANCHORS = 3;
    static constexpr size_t strides[3] = {8, 16, 32};

    YOLOv5Targets build_targets(
        size_t batch_size, size_t H, size_t W,
        size_t stride_idx,
        const std::vector<GroundTruth>& gt_batch) const;

    void compute_scale_loss(
        const Variable& pred,
        size_t stride_idx,
        const std::vector<GroundTruth>& gt_batch,
        Variable& box_loss,
        Variable& obj_loss,
        Variable& cls_loss) const;

public:
    YOLOv5Loss(size_t num_classes = 80,
               float box_weight = 0.05f,
               float obj_weight = 1.0f,
               float cls_weight = 0.5f,
               float anchor_threshold = 4.0f)
        : num_classes(num_classes),
          box_weight(box_weight),
          obj_weight(obj_weight),
          cls_weight(cls_weight),
          anchor_threshold(anchor_threshold) {}

    // All Variable ops -> loss.backward() propagates through model
    YOLOv5LossResult operator()(
        const std::vector<Variable>& preds,
        const std::vector<GroundTruth>& gt_batch);
};
