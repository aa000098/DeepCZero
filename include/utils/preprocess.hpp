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

/**
 * Letterbox 정보 (좌표 역변환에 사용)
 */
struct LetterboxInfo {
    float scale;
    int pad_left;
    int pad_top;
    int orig_width;
    int orig_height;
};

/**
 * YOLOv5용 이미지 전처리 (letterbox)
 * - RGB 로드
 * - 종횡비 유지하며 target_size에 맞춰 리사이즈
 * - 남는 영역은 gray(114) 패딩
 * - [0, 1] 정규화
 * - (1, 3, H, W) 텐서 반환
 */
tensor::Tensor<> preprocess_yolov5(const std::string& image_path,
                                    size_t target_size = 640,
                                    LetterboxInfo* info = nullptr);
