#include "container/loss/yolov5_loss.hpp"
#include "deepczero.hpp"

#include <cmath>
#include <algorithm>

static const std::vector<std::pair<float, float>>* ANCHOR_SETS[3] = {
    &YOLOV5_ANCHORS_P3, &YOLOV5_ANCHORS_P4, &YOLOV5_ANCHORS_P5
};

constexpr size_t YOLOv5Loss::strides[3];


YOLOv5Targets YOLOv5Loss::build_targets(
    size_t batch_size, size_t H, size_t W,
    size_t stride_idx,
    const std::vector<GroundTruth>& gt_batch) const {

    size_t stride = strides[stride_idx];
    const auto& anchors = *ANCHOR_SETS[stride_idx];

    YOLOv5Targets targets;
    targets.target_obj = Tensor<>({batch_size, NUM_ANCHORS, H, W}, 0.0f);

    std::vector<std::array<float, 4>> box_list;
    std::vector<float> cls_list;
    std::vector<size_t> flat_idx_list;

    for (size_t b = 0; b < batch_size; b++) {
        for (const auto& gt : gt_batch[b]) {
            float cls = gt[0];
            float cx_grid = gt[1] * static_cast<float>(W);
            float cy_grid = gt[2] * static_cast<float>(H);
            float w_pix = gt[3] * static_cast<float>(W * stride);
            float h_pix = gt[4] * static_cast<float>(H * stride);

            for (size_t a = 0; a < NUM_ANCHORS; a++) {
                float aw = anchors[a].first;
                float ah = anchors[a].second;

                float r_w = w_pix / aw;
                float r_h = h_pix / ah;
                float ratio = std::max({r_w, r_h, 1.0f / r_w, 1.0f / r_h});
                if (ratio > anchor_threshold) continue;

                int gx = std::clamp(static_cast<int>(cx_grid), 0, static_cast<int>(W) - 1);
                int gy = std::clamp(static_cast<int>(cy_grid), 0, static_cast<int>(H) - 1);

                targets.target_obj({b, a, static_cast<size_t>(gy), static_cast<size_t>(gx)}) = 1.0f;

                size_t flat_idx = b * (NUM_ANCHORS * H * W) +
                                  a * (H * W) +
                                  static_cast<size_t>(gy) * W +
                                  static_cast<size_t>(gx);
                flat_idx_list.push_back(flat_idx);

                float cx_offset = cx_grid - static_cast<float>(gx);
                float cy_offset = cy_grid - static_cast<float>(gy);
                float tw = w_pix / aw;
                float th = h_pix / ah;
                box_list.push_back({cx_offset, cy_offset, tw, th});
                cls_list.push_back(cls);
            }
        }
    }

    size_t N_pos = flat_idx_list.size();
    targets.flat_indices = flat_idx_list;

    if (N_pos > 0) {
        std::vector<float> box_data(N_pos * 4);
        for (size_t i = 0; i < N_pos; i++) {
            box_data[i * 4 + 0] = box_list[i][0];
            box_data[i * 4 + 1] = box_list[i][1];
            box_data[i * 4 + 2] = box_list[i][2];
            box_data[i * 4 + 3] = box_list[i][3];
        }
        targets.target_box = Tensor<>({N_pos, 4}, box_data);

        std::vector<float> cls_data(N_pos * num_classes, 0.0f);
        for (size_t i = 0; i < N_pos; i++) {
            size_t c = static_cast<size_t>(cls_list[i]);
            if (c < num_classes)
                cls_data[i * num_classes + c] = 1.0f;
        }
        targets.target_cls = Tensor<>({N_pos, num_classes}, cls_data);
    } else {
        targets.target_box = Tensor<>({0, 4});
        targets.target_cls = Tensor<>({0, num_classes});
    }

    return targets;
}


void YOLOv5Loss::compute_scale_loss(
    const Variable& pred,
    size_t stride_idx,
    const std::vector<GroundTruth>& gt_batch,
    Variable& box_loss,
    Variable& obj_loss,
    Variable& cls_loss) const {

    auto shape = pred.shape();
    size_t B = shape[0];
    size_t H = shape[2];
    size_t W = shape[3];
    size_t num_attrs = 5 + num_classes;
    size_t N_total = B * NUM_ANCHORS * H * W;

    // [B, A*(5+C), H, W] -> [B, A, (5+C), H, W] -> [B, A, H, W, (5+C)] -> [N, (5+C)]
    Variable x = reshape(pred, {B, NUM_ANCHORS, num_attrs, H, W});
    x = transpose(x, {0, 1, 3, 4, 2});
    x = reshape(x, {N_total, num_attrs});

    // Split along last axis using slice_axis (autograd tracks)
    Variable tx_ty = slice_axis(x, 1, 0, 2);       // [N, 2]
    Variable tw_th = slice_axis(x, 1, 2, 4);       // [N, 2]
    Variable obj_logits = slice_axis(x, 1, 4, 5);  // [N, 1]

    // Decode (differentiable via sigmoid)
    Variable pred_xy = sigmoid(tx_ty) * 2.0f - 0.5f;
    Variable pred_wh = (sigmoid(tw_th) * 2.0f) ^ 2.0f;

    // Build targets (label assignment, no gradient)
    auto targets = build_targets(B, H, W, stride_idx, gt_batch);
    size_t N_pos = targets.flat_indices.size();

    // Objectness loss (all cells)
    Variable obj_flat = reshape(obj_logits, {N_total});
    Variable obj_target(targets.target_obj.reshape({N_total}));
    obj_loss = obj_loss + binary_cross_entropy(obj_flat, obj_target);

    if (N_pos == 0) return;

    // Box loss (positive cells -> Gather -> CIoU Function)
    Variable pos_pred_xy = gather(pred_xy, targets.flat_indices);
    Variable pos_pred_wh = gather(pred_wh, targets.flat_indices);
    Variable pos_pred_box = concat({pos_pred_xy, pos_pred_wh}, 1);
    Variable target_box_var(targets.target_box);
    box_loss = box_loss + ciou_loss(pos_pred_box, target_box_var);

    // Classification loss (positive cells)
    if (num_classes > 1) {
        Variable cls_logits = slice_axis(x, 1, 5, num_attrs);
        Variable pos_cls_logits = gather(cls_logits, targets.flat_indices);
        Variable target_cls_var(targets.target_cls);

        Variable flat_cls_pred = reshape(pos_cls_logits, {N_pos * num_classes});
        Variable flat_cls_tgt = reshape(target_cls_var, {N_pos * num_classes});
        cls_loss = cls_loss + binary_cross_entropy(flat_cls_pred, flat_cls_tgt);
    }
}


YOLOv5LossResult YOLOv5Loss::operator()(
    const std::vector<Variable>& preds,
    const std::vector<GroundTruth>& gt_batch) {

    Variable box_loss(Tensor<>(1, 0.0f));
    Variable obj_loss(Tensor<>(1, 0.0f));
    Variable cls_loss(Tensor<>(1, 0.0f));

    for (size_t i = 0; i < 3 && i < preds.size(); i++) {
        compute_scale_loss(preds[i], i, gt_batch, box_loss, obj_loss, cls_loss);
    }

    Variable total = box_loss * box_weight + obj_loss * obj_weight + cls_loss * cls_weight;

    YOLOv5LossResult result;
    result.total_loss = total;
    Tensor<> box_cpu = box_loss.data().is_device() ? box_loss.data().cpu() : box_loss.data();
    Tensor<> obj_cpu = obj_loss.data().is_device() ? obj_loss.data().cpu() : obj_loss.data();
    Tensor<> cls_cpu = cls_loss.data().is_device() ? cls_loss.data().cpu() : cls_loss.data();
    result.box_loss_val = box_cpu.raw_data()[0];
    result.obj_loss_val = obj_cpu.raw_data()[0];
    result.cls_loss_val = cls_cpu.raw_data()[0];

    return result;
}
