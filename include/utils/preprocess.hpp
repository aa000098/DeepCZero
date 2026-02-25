#pragma once

#include "container/tensor/tensor_all.hpp"
#include <string>

/**
 * VGG16용 이미지 전처리
 * - RGB 변환
 * - 224x224 리사이즈
 * - BGR 변환 (채널 순서 반전)
 * - ImageNet mean 빼기: [103.939, 116.779, 123.68]
 * - (H, W, C) → (C, H, W) transpose
 * - batch dimension 추가: (1, C, H, W)
 */
tensor::Tensor<> preprocess_vgg16(const std::string& image_path);
