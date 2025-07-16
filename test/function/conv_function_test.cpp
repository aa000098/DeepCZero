#include "deepczero.hpp"

#include <cassert>
#include <iostream>

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
    test_im2col_forward();
    test_im2col_backward();
    test_col2im_forward();
    test_col2im_backward();

    return 0;
}
