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
#include <cstdio>

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

// 8x8 bitmap font (printable ASCII 32-126, CP437 style)
static const unsigned char FONT_8X8[95][8] = {
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00}, // 32: space
    {0x18,0x3C,0x3C,0x18,0x18,0x00,0x18,0x00}, // 33: !
    {0x36,0x36,0x14,0x00,0x00,0x00,0x00,0x00}, // 34: "
    {0x36,0x36,0x7F,0x36,0x7F,0x36,0x36,0x00}, // 35: #
    {0x0C,0x3E,0x03,0x1E,0x30,0x1F,0x0C,0x00}, // 36: $
    {0x00,0x63,0x33,0x18,0x0C,0x66,0x63,0x00}, // 37: %
    {0x1C,0x36,0x1C,0x6E,0x3B,0x33,0x6E,0x00}, // 38: &
    {0x06,0x06,0x03,0x00,0x00,0x00,0x00,0x00}, // 39: '
    {0x18,0x0C,0x06,0x06,0x06,0x0C,0x18,0x00}, // 40: (
    {0x06,0x0C,0x18,0x18,0x18,0x0C,0x06,0x00}, // 41: )
    {0x00,0x66,0x3C,0xFF,0x3C,0x66,0x00,0x00}, // 42: *
    {0x00,0x0C,0x0C,0x3F,0x0C,0x0C,0x00,0x00}, // 43: +
    {0x00,0x00,0x00,0x00,0x00,0x0C,0x0C,0x06}, // 44: ,
    {0x00,0x00,0x00,0x3F,0x00,0x00,0x00,0x00}, // 45: -
    {0x00,0x00,0x00,0x00,0x00,0x0C,0x0C,0x00}, // 46: .
    {0x60,0x30,0x18,0x0C,0x06,0x03,0x01,0x00}, // 47: /
    {0x3E,0x63,0x73,0x7B,0x6F,0x67,0x3E,0x00}, // 48: 0
    {0x0C,0x0E,0x0C,0x0C,0x0C,0x0C,0x3F,0x00}, // 49: 1
    {0x1E,0x33,0x30,0x1C,0x06,0x33,0x3F,0x00}, // 50: 2
    {0x1E,0x33,0x30,0x1C,0x30,0x33,0x1E,0x00}, // 51: 3
    {0x38,0x3C,0x36,0x33,0x7F,0x30,0x78,0x00}, // 52: 4
    {0x3F,0x03,0x1F,0x30,0x30,0x33,0x1E,0x00}, // 53: 5
    {0x1C,0x06,0x03,0x1F,0x33,0x33,0x1E,0x00}, // 54: 6
    {0x3F,0x33,0x30,0x18,0x0C,0x0C,0x0C,0x00}, // 55: 7
    {0x1E,0x33,0x33,0x1E,0x33,0x33,0x1E,0x00}, // 56: 8
    {0x1E,0x33,0x33,0x3E,0x30,0x18,0x0E,0x00}, // 57: 9
    {0x00,0x0C,0x0C,0x00,0x00,0x0C,0x0C,0x00}, // 58: :
    {0x00,0x0C,0x0C,0x00,0x00,0x0C,0x0C,0x06}, // 59: ;
    {0x18,0x0C,0x06,0x03,0x06,0x0C,0x18,0x00}, // 60: <
    {0x00,0x00,0x3F,0x00,0x00,0x3F,0x00,0x00}, // 61: =
    {0x06,0x0C,0x18,0x30,0x18,0x0C,0x06,0x00}, // 62: >
    {0x1E,0x33,0x30,0x18,0x0C,0x00,0x0C,0x00}, // 63: ?
    {0x3E,0x63,0x7B,0x7B,0x7B,0x03,0x1E,0x00}, // 64: @
    {0x0C,0x1E,0x33,0x33,0x3F,0x33,0x33,0x00}, // 65: A
    {0x3F,0x66,0x66,0x3E,0x66,0x66,0x3F,0x00}, // 66: B
    {0x3C,0x66,0x03,0x03,0x03,0x66,0x3C,0x00}, // 67: C
    {0x1F,0x36,0x66,0x66,0x66,0x36,0x1F,0x00}, // 68: D
    {0x7F,0x46,0x16,0x1E,0x16,0x46,0x7F,0x00}, // 69: E
    {0x7F,0x46,0x16,0x1E,0x16,0x06,0x0F,0x00}, // 70: F
    {0x3C,0x66,0x03,0x03,0x73,0x66,0x7C,0x00}, // 71: G
    {0x33,0x33,0x33,0x3F,0x33,0x33,0x33,0x00}, // 72: H
    {0x1E,0x0C,0x0C,0x0C,0x0C,0x0C,0x1E,0x00}, // 73: I
    {0x78,0x30,0x30,0x30,0x33,0x33,0x1E,0x00}, // 74: J
    {0x67,0x66,0x36,0x1E,0x36,0x66,0x67,0x00}, // 75: K
    {0x0F,0x06,0x06,0x06,0x46,0x66,0x7F,0x00}, // 76: L
    {0x63,0x77,0x7F,0x7F,0x6B,0x63,0x63,0x00}, // 77: M
    {0x63,0x67,0x6F,0x7B,0x73,0x63,0x63,0x00}, // 78: N
    {0x1C,0x36,0x63,0x63,0x63,0x36,0x1C,0x00}, // 79: O
    {0x3F,0x66,0x66,0x3E,0x06,0x06,0x0F,0x00}, // 80: P
    {0x1E,0x33,0x33,0x33,0x3B,0x1E,0x38,0x00}, // 81: Q
    {0x3F,0x66,0x66,0x3E,0x36,0x66,0x67,0x00}, // 82: R
    {0x1E,0x33,0x07,0x0E,0x38,0x33,0x1E,0x00}, // 83: S
    {0x3F,0x2D,0x0C,0x0C,0x0C,0x0C,0x1E,0x00}, // 84: T
    {0x33,0x33,0x33,0x33,0x33,0x33,0x3F,0x00}, // 85: U
    {0x33,0x33,0x33,0x33,0x33,0x1E,0x0C,0x00}, // 86: V
    {0x63,0x63,0x63,0x6B,0x7F,0x77,0x63,0x00}, // 87: W
    {0x63,0x63,0x36,0x1C,0x1C,0x36,0x63,0x00}, // 88: X
    {0x33,0x33,0x33,0x1E,0x0C,0x0C,0x1E,0x00}, // 89: Y
    {0x7F,0x63,0x31,0x18,0x4C,0x66,0x7F,0x00}, // 90: Z
    {0x1E,0x06,0x06,0x06,0x06,0x06,0x1E,0x00}, // 91: [
    {0x03,0x06,0x0C,0x18,0x30,0x60,0x40,0x00}, // 92: backslash
    {0x1E,0x18,0x18,0x18,0x18,0x18,0x1E,0x00}, // 93: ]
    {0x08,0x1C,0x36,0x63,0x00,0x00,0x00,0x00}, // 94: ^
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xFF}, // 95: _
    {0x0C,0x0C,0x18,0x00,0x00,0x00,0x00,0x00}, // 96: `
    {0x00,0x00,0x1E,0x30,0x3E,0x33,0x6E,0x00}, // 97: a
    {0x07,0x06,0x06,0x3E,0x66,0x66,0x3B,0x00}, // 98: b
    {0x00,0x00,0x1E,0x33,0x03,0x33,0x1E,0x00}, // 99: c
    {0x38,0x30,0x30,0x3E,0x33,0x33,0x6E,0x00}, // 100: d
    {0x00,0x00,0x1E,0x33,0x3F,0x03,0x1E,0x00}, // 101: e
    {0x1C,0x36,0x06,0x0F,0x06,0x06,0x0F,0x00}, // 102: f
    {0x00,0x00,0x6E,0x33,0x33,0x3E,0x30,0x1F}, // 103: g
    {0x07,0x06,0x36,0x6E,0x66,0x66,0x67,0x00}, // 104: h
    {0x0C,0x00,0x0E,0x0C,0x0C,0x0C,0x1E,0x00}, // 105: i
    {0x30,0x00,0x30,0x30,0x30,0x33,0x33,0x1E}, // 106: j
    {0x07,0x06,0x66,0x36,0x1E,0x36,0x67,0x00}, // 107: k
    {0x0E,0x0C,0x0C,0x0C,0x0C,0x0C,0x1E,0x00}, // 108: l
    {0x00,0x00,0x33,0x7F,0x7F,0x6B,0x63,0x00}, // 109: m
    {0x00,0x00,0x1F,0x33,0x33,0x33,0x33,0x00}, // 110: n
    {0x00,0x00,0x1E,0x33,0x33,0x33,0x1E,0x00}, // 111: o
    {0x00,0x00,0x3B,0x66,0x66,0x3E,0x06,0x0F}, // 112: p
    {0x00,0x00,0x6E,0x33,0x33,0x3E,0x30,0x78}, // 113: q
    {0x00,0x00,0x3B,0x6E,0x66,0x06,0x0F,0x00}, // 114: r
    {0x00,0x00,0x3E,0x03,0x1E,0x30,0x1F,0x00}, // 115: s
    {0x08,0x0C,0x3E,0x0C,0x0C,0x2C,0x18,0x00}, // 116: t
    {0x00,0x00,0x33,0x33,0x33,0x33,0x6E,0x00}, // 117: u
    {0x00,0x00,0x33,0x33,0x33,0x1E,0x0C,0x00}, // 118: v
    {0x00,0x00,0x63,0x6B,0x7F,0x7F,0x36,0x00}, // 119: w
    {0x00,0x00,0x63,0x36,0x1C,0x36,0x63,0x00}, // 120: x
    {0x00,0x00,0x33,0x33,0x33,0x3E,0x30,0x1F}, // 121: y
    {0x00,0x00,0x3F,0x19,0x0C,0x26,0x3F,0x00}, // 122: z
    {0x38,0x0C,0x0C,0x07,0x0C,0x0C,0x38,0x00}, // 123: {
    {0x18,0x18,0x18,0x00,0x18,0x18,0x18,0x00}, // 124: |
    {0x07,0x0C,0x0C,0x38,0x0C,0x0C,0x07,0x00}, // 125: }
    {0x6E,0x3B,0x00,0x00,0x00,0x00,0x00,0x00}, // 126: ~
};

static void draw_filled_rect(unsigned char* img, int img_w, int img_h,
                              int x1, int y1, int x2, int y2,
                              unsigned char r, unsigned char g, unsigned char b) {
    x1 = std::max(0, std::min(x1, img_w - 1));
    y1 = std::max(0, std::min(y1, img_h - 1));
    x2 = std::max(0, std::min(x2, img_w - 1));
    y2 = std::max(0, std::min(y2, img_h - 1));
    for (int y = y1; y <= y2; ++y) {
        for (int x = x1; x <= x2; ++x) {
            img[(y * img_w + x) * 3 + 0] = r;
            img[(y * img_w + x) * 3 + 1] = g;
            img[(y * img_w + x) * 3 + 2] = b;
        }
    }
}

static void draw_char(unsigned char* img, int img_w, int img_h,
                      int x0, int y0, char ch,
                      unsigned char r, unsigned char g, unsigned char b) {
    int idx = static_cast<int>(ch) - 32;
    if (idx < 0 || idx >= 95) return;
    for (int row = 0; row < 8; ++row) {
        unsigned char bits = FONT_8X8[idx][row];
        for (int col = 0; col < 8; ++col) {
            if (bits & (1 << col)) {
                int px = x0 + col;
                int py = y0 + row;
                if (px >= 0 && px < img_w && py >= 0 && py < img_h) {
                    img[(py * img_w + px) * 3 + 0] = r;
                    img[(py * img_w + px) * 3 + 1] = g;
                    img[(py * img_w + px) * 3 + 2] = b;
                }
            }
        }
    }
}

static void draw_text(unsigned char* img, int img_w, int img_h,
                      int x0, int y0, const std::string& text,
                      unsigned char r, unsigned char g, unsigned char b) {
    for (size_t i = 0; i < text.size(); ++i) {
        draw_char(img, img_w, img_h, x0 + static_cast<int>(i) * 8, y0, text[i], r, g, b);
    }
}

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

        int bx1 = static_cast<int>(det.x1);
        int by1 = static_cast<int>(det.y1);
        int bx2 = static_cast<int>(det.x2);
        int by2 = static_cast<int>(det.y2);

        draw_rect(img, w, h, bx1, by1, bx2, by2, r, g, b);

        // Label: "classname 0.XX"
        std::string cls_name = (det.class_id >= 0 &&
            det.class_id < static_cast<int>(COCO_CLASSES.size()))
            ? COCO_CLASSES[det.class_id] : "unknown";
        char conf_buf[8];
        std::snprintf(conf_buf, sizeof(conf_buf), "%.2f", det.confidence);
        std::string label = cls_name + " " + conf_buf;

        const int font_h = 8;
        const int pad = 3;
        int label_w = static_cast<int>(label.size()) * 8 + pad * 2;
        int label_h = font_h + pad * 2;

        int lx = bx1;
        int ly = by1 - label_h;
        if (ly < 0) ly = by1;
        if (lx + label_w > w) lx = w - label_w;
        if (lx < 0) lx = 0;

        draw_filled_rect(img, w, h, lx, ly, lx + label_w - 1, ly + label_h - 1, r, g, b);

        // Pick text color based on background luminance
        float lum = 0.299f * r + 0.587f * g + 0.114f * b;
        unsigned char tr = (lum > 128) ? 0 : 255;
        unsigned char tg = (lum > 128) ? 0 : 255;
        unsigned char tb = (lum > 128) ? 0 : 255;
        draw_text(img, w, h, lx + pad, ly + pad, label, tr, tg, tb);
    }

    stbi_write_jpg(output_path.c_str(), w, h, 3, img, 95);
    stbi_image_free(img);
}
