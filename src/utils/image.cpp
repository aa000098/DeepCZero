#include "utils/image.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

#include <stdexcept>
#include <iostream>

tensor::Tensor<> preprocess_vgg16(const std::string& image_path) {
    // 1. 이미지 로드 (RGB 강제)
    int width, height, channels;
    unsigned char* img_data = stbi_load(image_path.c_str(), &width, &height, &channels, 3);

    if (!img_data) {
        throw std::runtime_error("Failed to load image: " + image_path);
    }

    std::cerr << "[preprocess] Loaded image: " << width << "x" << height << " (channels: " << channels << ")" << std::endl;

    // 2. 224x224 리사이즈
    const int target_size = 224;
    unsigned char* resized_data = new unsigned char[target_size * target_size * 3];

    stbir_resize_uint8_linear(
        img_data, width, height, 0,
        resized_data, target_size, target_size, 0,
        STBIR_RGB
    );

    stbi_image_free(img_data);

    // 3. Tensor 생성 및 전처리
    // (C, H, W) = (3, 224, 224)
    tensor::Tensor<> result({3, target_size, target_size});

    // ImageNet mean (BGR 순서)
    const float mean_b = 103.939f;
    const float mean_g = 116.779f;
    const float mean_r = 123.68f;

    for (int h = 0; h < target_size; ++h) {
        for (int w = 0; w < target_size; ++w) {
            int idx = (h * target_size + w) * 3;

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
