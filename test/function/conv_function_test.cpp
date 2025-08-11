#include "deepczero.hpp"

#include <cassert>
#include <iostream>

void test_conv2d_forward_backward() {
    std::cout << "[Test] Conv2d forward/backward" << std::endl;

    // 입력 데이터 초기화
	// [N, C, H, W]
    Tensor<> x_tensor({1, 3, 5, 5});
	// [OC, C, KH, KW]
    Tensor<> W_tensor({3, 3, 3, 3});
    Tensor<> b_tensor({3});

    for (size_t i = 0; i < x_tensor.size(); ++i) x_tensor.raw_data()[i] = static_cast<float>(i + 1);
    for (size_t i = 0; i < W_tensor.size(); ++i) W_tensor.raw_data()[i] = static_cast<float>(0.01 * i);
    for (size_t i = 0; i < b_tensor.size(); ++i) b_tensor.raw_data()[i] = static_cast<float>(0.1 * i);

    // Variable 래핑
    Variable x(x_tensor);
    Variable W(W_tensor);
    Variable b(b_tensor);

    // Forward
	// [N, OC, OH, OW]
    Variable y = conv2d(x, W, b, {1, 1}, {1, 1});
    const Tensor<> &y_data = y.data();

	std::cout << "[Y data]" << std::endl;
	y_data.show();

    // Shape 검증
    assert(y_data.get_shape() == std::vector<size_t>({1, 3, 5, 5}));
    std::cout << "Conv2d forward shape passed." << std::endl;

    // ===== Forward expected 값으로 검증 (NCHW 플랫 순서) =====
    std::vector<float> expv = {
      // ch 0
      70.68f,104.79f,107.40f,110.01f, 72.24f,
      105.39f,155.70f,159.21f,162.72f,106.47f,
      117.54f,173.25f,176.76f,180.27f,117.72f,
      129.69f,190.80f,194.31f,197.82f,128.97f,
       81.84f,119.91f,121.98f,124.05f, 80.52f,
      // ch 1
      164.74f,248.26f,255.73f,263.20f,176.02f,
      258.58f,389.08f,399.88f,410.68f,274.24f,
      295.03f,443.08f,453.88f,464.68f,309.79f,
      331.48f,497.08f,507.88f,518.68f,345.34f,
      224.50f,336.28f,343.21f,350.14f,232.90f,
      // ch 2
      258.80f,391.73f,404.06f,416.39f,279.80f,
      411.77f,622.46f,640.55f,658.64f,442.01f,
      472.52f,712.91f,731.00f,749.09f,501.86f,
      533.27f,803.36f,821.45f,839.54f,561.71f,
      367.16f,552.65f,564.44f,576.23f,385.28f
    };

    double max_abs_diff = 0.0, mean_abs_diff = 0.0;
    for (size_t i = 0; i < y_data.size(); ++i) {
        double d = std::abs((double)y_data.data()[i] - (double)expv[i]);
        max_abs_diff = std::max(max_abs_diff, d);
        mean_abs_diff += d;
    }
    mean_abs_diff /= (double)y_data.size();
    std::cout << "Forward max abs diff: " << max_abs_diff
              << ", mean abs diff: " << mean_abs_diff << "\n";
    assert(max_abs_diff < 1e-4);

	//값이 각 출력 채널에 영향 미쳤는지 평균으로 확인
    for (size_t c = 0; c < 3; ++c) {
        float sum = 0.0f;
        for (size_t i = 0; i < 25; ++i) {
            sum += y_data.data()[c * 25 + i];
        }
        float avg_val = sum / 25.0f;
        float expected_bias = b_tensor.raw_data()[c];
        assert(std::abs(avg_val - expected_bias) > 0.01f);  // b가 반영되었는지만 확인
    }
    std::cout << "Conv2d forward value sanity check passed." << std::endl;

    // Backward (모든 위치에 gradient 1.0으로 설정)
    y.backward();

    // Gradient shape 확인
    assert(x.grad().shape() == x.shape());
    assert(W.grad().shape() == W.shape());
    assert(b.grad().shape() == b.shape());
    std::cout << "Conv2d backward shape check passed." << std::endl;

    // Bias gradient 값 검증: 각 출력 채널에 대해 1.0 × 25 = 25.0이 누적되어야 함
    for (size_t i = 0; i < 3; ++i) {
        float expected = 25.0f;
        float actual = b.grad().data().raw_data()[i];
        assert(std::abs(actual - expected) < 1e-3);
    }

    std::cout << "Conv2d backward db value check passed." << std::endl;
    std::cout << "✅ Conv2d forward/backward full test passed.\n" << std::endl;
}

void test_im2col_forward() {
    using T = float;

    // 입력 이미지 (N, C, H, W) 형태
    Tensor<T> img({1, 1, 4, 4});
    for (size_t i = 0; i < 16; ++i)
        img.raw_data()[i] = static_cast<T>(i + 1);

    Variable x(img);

    // Forward 수행
    Variable y = im2col(x, {3, 3}, {1, 1}, {1, 1}, false);

    // 예상된 shape: {N, C, KH, KW, OH, OW} => {1, 1, 3, 3, 4, 4}
    std::vector<size_t> expected_shape = {1, 1, 3, 3, 4, 4};
    assert(y.data().get_shape() == expected_shape);

    std::cout << "[Im2col Forward Test Passed]" << std::endl;
}

void test_im2col_backward() {
    using T = float;

    // 원본 입력 텐서
    Tensor<T> img({1, 1, 4, 4}, 1.0f);
    Variable x(img);

    // Forward 한 번 호출하여 input_shape 설정
    Variable y = im2col(x, {3, 3}, {1, 1}, {1, 1}, false);

    // Backward 수행
    y.backward();

	x.grad().show();
    // 예상된 shape: 입력 텐서와 동일 {1, 1, 4, 4}
    std::vector<size_t> expected_shape = {1, 1, 4, 4};
    assert(x.grad().data().get_shape() == expected_shape);

    std::cout << "[Im2col Backward Test Passed]" << std::endl;
}

void test_col2im_forward() {
    using T = float;

    // 입력 텐서 (im2col의 출력과 동일한 shape)
    Tensor<T> col_data({1, 1, 3, 3, 4, 4}, 1.0f);
    Variable x(col_data);

    // Forward 수행
    Variable y = col2im(x, {1, 1, 4, 4}, {3, 3}, {1, 1}, {1, 1}, false);

	// 예상된 shape: 입력 이미지 shape {1, 1, 4, 4}
    std::vector<size_t> expected_shape = {1, 1, 4, 4};
    assert(y.data().get_shape() == expected_shape);

    std::cout << "[Col2im Forward Test Passed]" << std::endl;
}


void test_col2im_backward() {
    using T = float;

    // 입력 텐서 (im2col 결과 형태)
    Tensor<T> col_data({1, 1, 3, 3, 4, 4}, 1.0f);
    Variable x(col_data);

    // Forward 한 번 호출하여 input_shape 설정
    Variable y = col2im(x, {1, 1, 4, 4}, {3, 3}, {1, 1}, {1, 1}, false);

    // Backward 수행
    y.backward();

	x.grad().show();
    // 예상된 shape: 입력 텐서 shape과 동일 {1, 1, 3, 3, 4, 4}
    std::vector<size_t> expected_shape = {1, 1, 3, 3, 4, 4};
    assert(x.grad().data().get_shape() == expected_shape);

    std::cout << "[Col2im Backward Test Passed]" << std::endl;
}

int main() {
	test_conv2d_forward_backward();
    test_im2col_forward();
    test_im2col_backward();
    test_col2im_forward();
    test_col2im_backward();

    return 0;
}
