#include "utils/postprocess.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#include "stb_image_write.h"
#pragma GCC diagnostic pop

#include "stb_image.h"

#include <cmath>
#include <algorithm>
#include <iostream>

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

static float iou(const Detection& a, const Detection& b) {
    float inter_x1 = std::max(a.x1, b.x1);
    float inter_y1 = std::max(a.y1, b.y1);
    float inter_x2 = std::min(a.x2, b.x2);
    float inter_y2 = std::min(a.y2, b.y2);

    float inter_area = std::max(0.0f, inter_x2 - inter_x1) *
                       std::max(0.0f, inter_y2 - inter_y1);
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);

    return inter_area / (area_a + area_b - inter_area + 1e-6f);
}

static std::vector<Detection> decode_single_head(
    const tensor::Tensor<>& output,
    const std::vector<std::pair<float, float>>& anchors,
    size_t stride,
    size_t num_classes,
    float conf_threshold) {

    std::vector<Detection> dets;
    auto shape = output.get_shape();
    // shape: [1, num_anchors*(5+num_classes), H, W]
    size_t H = shape[2];
    size_t W = shape[3];
    size_t num_attrs = 5 + num_classes;
    size_t num_anchors = anchors.size();

    const auto& data = output.raw_data();

    for (size_t a = 0; a < num_anchors; ++a) {
        for (size_t gy = 0; gy < H; ++gy) {
            for (size_t gx = 0; gx < W; ++gx) {
                // Index into [1, C, H, W]: batch=0, channel=c, row=gy, col=gx
                // offset = c * H * W + gy * W + gx
                size_t base = (a * num_attrs) * H * W + gy * W + gx;

                float obj_raw = data[base + 4 * H * W];
                float obj_score = sigmoid(obj_raw);

                if (obj_score < conf_threshold * 0.5f) continue;

                // Find best class
                float max_cls_score = -1e9f;
                int best_cls = 0;
                for (size_t c = 0; c < num_classes; ++c) {
                    float cls_raw = data[base + (5 + c) * H * W];
                    float cls_score = sigmoid(cls_raw);
                    if (cls_score > max_cls_score) {
                        max_cls_score = cls_score;
                        best_cls = static_cast<int>(c);
                    }
                }

                float confidence = obj_score * max_cls_score;
                if (confidence < conf_threshold) continue;

                // Decode bbox
                float tx = data[base + 0 * H * W];
                float ty = data[base + 1 * H * W];
                float tw = data[base + 2 * H * W];
                float th = data[base + 3 * H * W];

                float cx = (sigmoid(tx) * 2.0f - 0.5f + gx) * stride;
                float cy = (sigmoid(ty) * 2.0f - 0.5f + gy) * stride;
                float w = std::pow(sigmoid(tw) * 2.0f, 2) * anchors[a].first;
                float h = std::pow(sigmoid(th) * 2.0f, 2) * anchors[a].second;

                Detection det;
                det.x1 = cx - w / 2.0f;
                det.y1 = cy - h / 2.0f;
                det.x2 = cx + w / 2.0f;
                det.y2 = cy + h / 2.0f;
                det.confidence = confidence;
                det.class_id = best_cls;

                dets.push_back(det);
            }
        }
    }

    return dets;
}

std::vector<Detection> decode_yolov5_outputs(
    const std::vector<Variable>& outputs,
    size_t num_classes,
    float conf_threshold) {

    static const size_t strides[] = {8, 16, 32};
    static const std::vector<std::pair<float, float>>* anchor_sets[] = {
        &YOLOV5_ANCHORS_P3, &YOLOV5_ANCHORS_P4, &YOLOV5_ANCHORS_P5
    };

    std::vector<Detection> all_dets;

    for (size_t i = 0; i < 3 && i < outputs.size(); ++i) {
        tensor::Tensor<> t = outputs[i].data().contiguous();
        auto dets = decode_single_head(t, *anchor_sets[i], strides[i],
                                       num_classes, conf_threshold);
        all_dets.insert(all_dets.end(), dets.begin(), dets.end());
    }

    return all_dets;
}

std::vector<Detection> nms(
    std::vector<Detection>& detections,
    float iou_threshold) {

    // Sort by confidence descending
    std::sort(detections.begin(), detections.end(),
        [](const Detection& a, const Detection& b) {
            return a.confidence > b.confidence;
        });

    std::vector<bool> suppressed(detections.size(), false);
    std::vector<Detection> result;

    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i]) continue;
        result.push_back(detections[i]);

        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (suppressed[j]) continue;
            if (detections[i].class_id != detections[j].class_id) continue;

            if (iou(detections[i], detections[j]) > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }

    return result;
}

void rescale_detections(
    std::vector<Detection>& detections,
    const LetterboxInfo& info) {

    for (auto& d : detections) {
        d.x1 = (d.x1 - info.pad_left) / info.scale;
        d.y1 = (d.y1 - info.pad_top) / info.scale;
        d.x2 = (d.x2 - info.pad_left) / info.scale;
        d.y2 = (d.y2 - info.pad_top) / info.scale;

        d.x1 = std::max(0.0f, std::min(d.x1, static_cast<float>(info.orig_width)));
        d.y1 = std::max(0.0f, std::min(d.y1, static_cast<float>(info.orig_height)));
        d.x2 = std::max(0.0f, std::min(d.x2, static_cast<float>(info.orig_width)));
        d.y2 = std::max(0.0f, std::min(d.y2, static_cast<float>(info.orig_height)));
    }
}

// Simple color palette for drawing (12 colors)
static const unsigned char COLORS[][3] = {
    {255, 56, 56},   {255, 157, 151}, {255, 112, 31},  {255, 178, 29},
    {207, 210, 49},  {72, 249, 10},   {146, 204, 23},  {61, 219, 134},
    {26, 147, 52},   {0, 212, 187},   {44, 153, 168},  {0, 194, 255},
};

static void draw_rect(unsigned char* img, int img_w, int img_h,
                      int x1, int y1, int x2, int y2,
                      unsigned char r, unsigned char g, unsigned char b,
                      int thickness = 3) {
    x1 = std::max(0, std::min(x1, img_w - 1));
    y1 = std::max(0, std::min(y1, img_h - 1));
    x2 = std::max(0, std::min(x2, img_w - 1));
    y2 = std::max(0, std::min(y2, img_h - 1));

    for (int t = 0; t < thickness; ++t) {
        // Top edge
        for (int x = x1; x <= x2; ++x) {
            int y = y1 + t;
            if (y >= 0 && y < img_h) {
                img[(y * img_w + x) * 3 + 0] = r;
                img[(y * img_w + x) * 3 + 1] = g;
                img[(y * img_w + x) * 3 + 2] = b;
            }
        }
        // Bottom edge
        for (int x = x1; x <= x2; ++x) {
            int y = y2 - t;
            if (y >= 0 && y < img_h) {
                img[(y * img_w + x) * 3 + 0] = r;
                img[(y * img_w + x) * 3 + 1] = g;
                img[(y * img_w + x) * 3 + 2] = b;
            }
        }
        // Left edge
        for (int y = y1; y <= y2; ++y) {
            int x = x1 + t;
            if (x >= 0 && x < img_w) {
                img[(y * img_w + x) * 3 + 0] = r;
                img[(y * img_w + x) * 3 + 1] = g;
                img[(y * img_w + x) * 3 + 2] = b;
            }
        }
        // Right edge
        for (int y = y1; y <= y2; ++y) {
            int x = x2 - t;
            if (x >= 0 && x < img_w) {
                img[(y * img_w + x) * 3 + 0] = r;
                img[(y * img_w + x) * 3 + 1] = g;
                img[(y * img_w + x) * 3 + 2] = b;
            }
        }
    }
}

void draw_detections(
    const std::string& input_image_path,
    const std::vector<Detection>& detections,
    const std::string& output_path) {

    int w, h, ch;
    unsigned char* img = stbi_load(input_image_path.c_str(), &w, &h, &ch, 3);
    if (!img) {
        std::cerr << "Failed to load image for drawing: " << input_image_path << std::endl;
        return;
    }

    for (const auto& det : detections) {
        int color_idx = det.class_id % 12;
        unsigned char r = COLORS[color_idx][0];
        unsigned char g = COLORS[color_idx][1];
        unsigned char b = COLORS[color_idx][2];

        draw_rect(img, w, h,
                  static_cast<int>(det.x1), static_cast<int>(det.y1),
                  static_cast<int>(det.x2), static_cast<int>(det.y2),
                  r, g, b);
    }

    stbi_write_jpg(output_path.c_str(), w, h, 3, img, 95);
    stbi_image_free(img);
}
