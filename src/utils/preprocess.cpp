#include "utils/preprocess.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

#include <stdexcept>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <cmath>

tensor::Tensor<> preprocess_vgg16(const std::string& image_path) {
    // 1. 이미지 로드 (RGB 강제)
    int width, height, channels;
    unsigned char* img_data = stbi_load(image_path.c_str(), &width, &height, &channels, 3);

    if (!img_data) {
        throw std::runtime_error("Failed to load image: " + image_path);
    }

    std::cerr << "[preprocess] Loaded image: " << width << "x" << height << " (channels: " << channels << ")" << std::endl;

    // 2. 224x224 리사이즈
    const size_t target_size = 224;
    unsigned char* resized_data = new unsigned char[target_size * target_size * 3];

    // stb의 easy API는 다운샘플링에 Mitchell 필터 사용 (PIL BICUBIC과 유사)
    STBIR_RESIZE resize;
    stbir_resize_init(&resize,
        img_data, width, height, 0,
        resized_data, target_size, target_size, 0,
        STBIR_RGB, STBIR_TYPE_UINT8);

    // Catmull-Rom 필터 사용 (PIL BICUBIC과 유사)
    stbir_set_filters(&resize, STBIR_FILTER_CATMULLROM, STBIR_FILTER_CATMULLROM);

    stbir_resize_extended(&resize);

    stbi_image_free(img_data);

    // 3. Tensor 생성 및 전처리
    // (C, H, W) = (3, 224, 224)
    tensor::Tensor<> result({3, target_size, target_size});

    // ImageNet mean (BGR 순서)
    const float mean_b = 103.939f;
    const float mean_g = 116.779f;
    const float mean_r = 123.68f;

    for (size_t h = 0; h < target_size; ++h) {
        for (size_t w = 0; w < target_size; ++w) {
            size_t idx = (h * target_size + w) * 3;

            // RGB → BGR 변환 및 mean 빼기
            float r = static_cast<float>(resized_data[idx + 0]);
            float g = static_cast<float>(resized_data[idx + 1]);
            float b = static_cast<float>(resized_data[idx + 2]);

            // (H, W, C) → (C, H, W) transpose
            // 채널 순서: BGR
            result({0, h, w}) = b - mean_b;
            result({1, h, w}) = g - mean_g;
            result({2, h, w}) = r - mean_r;
        }
    }

    delete[] resized_data;

    // 4. batch dimension 추가: (C, H, W) → (1, C, H, W)
    result = result.reshape({1, 3, target_size, target_size});

    std::cerr << "[preprocess] Output shape: (1, 3, 224, 224)" << std::endl;

    return result;
}

tensor::Tensor<> preprocess_yolov5(const std::string& image_path,
                                    size_t target_size,
                                    LetterboxInfo* info) {
    // 1. 이미지 로드 (RGB)
    int width, height, channels;
    unsigned char* img_data = stbi_load(image_path.c_str(), &width, &height, &channels, 3);

    if (!img_data) {
        throw std::runtime_error("Failed to load image: " + image_path);
    }

    // 2. Letterbox: 종횡비 유지하며 리사이즈
    float scale = std::min(static_cast<float>(target_size) / width,
                           static_cast<float>(target_size) / height);
    int new_w = static_cast<int>(std::round(width * scale));
    int new_h = static_cast<int>(std::round(height * scale));

    // 리사이즈
    unsigned char* resized = new unsigned char[new_w * new_h * 3];
    STBIR_RESIZE resize;
    stbir_resize_init(&resize,
        img_data, width, height, 0,
        resized, new_w, new_h, 0,
        STBIR_RGB, STBIR_TYPE_UINT8);
    stbir_set_filters(&resize, STBIR_FILTER_CATMULLROM, STBIR_FILTER_CATMULLROM);
    stbir_resize_extended(&resize);

    stbi_image_free(img_data);

    // 3. 패딩 (gray 114)
    int pad_left = (static_cast<int>(target_size) - new_w) / 2;
    int pad_top  = (static_cast<int>(target_size) - new_h) / 2;

    size_t buf_size = target_size * target_size * 3;
    unsigned char* padded = new unsigned char[buf_size];
    std::memset(padded, 114, buf_size);  // gray fill

    for (int y = 0; y < new_h; ++y) {
        for (int x = 0; x < new_w; ++x) {
            int src_idx = (y * new_w + x) * 3;
            int dst_idx = ((pad_top + y) * static_cast<int>(target_size) + (pad_left + x)) * 3;
            padded[dst_idx + 0] = resized[src_idx + 0];
            padded[dst_idx + 1] = resized[src_idx + 1];
            padded[dst_idx + 2] = resized[src_idx + 2];
        }
    }

    delete[] resized;

    // 4. HWC uint8 → CHW float [0,1]
    tensor::Tensor<> result({3, target_size, target_size});
    for (size_t h = 0; h < target_size; ++h) {
        for (size_t w = 0; w < target_size; ++w) {
            size_t idx = (h * target_size + w) * 3;
            result({0, h, w}) = padded[idx + 0] / 255.0f;  // R
            result({1, h, w}) = padded[idx + 1] / 255.0f;  // G
            result({2, h, w}) = padded[idx + 2] / 255.0f;  // B
        }
    }

    delete[] padded;

    result = result.reshape({1, 3, target_size, target_size});

    // 5. Letterbox 정보 저장
    if (info) {
        info->scale = scale;
        info->pad_left = pad_left;
        info->pad_top = pad_top;
        info->orig_width = width;
        info->orig_height = height;
    }

    return result;
}
