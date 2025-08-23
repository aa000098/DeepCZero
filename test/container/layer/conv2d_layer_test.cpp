#include "deepczero.hpp"

#include <cassert>
#include <iostream>
#include <cmath>

void test_conv2d_forward() {
    using T = float;

    // 입력 데이터 (batch=1, in_channels=1, H=4, W=4)
    Tensor<T> x_data({1, 1, 4, 4}, 1.0f);
    Variable x(x_data);

    // Conv2d 레이어: out_channels=2, kernel_size=3x3, stride=1, no_bias=false, pad=1
    layer::Conv2d conv(2, {3, 3}, {1, 1}, {1, 1}, false, 1);

    // Forward 수행
    Variable y = conv.forward({x});

    // 예상 출력 shape: {batch=1, out_channels=2, H=4, W=4}
    std::vector<size_t> expected_shape = {1, 2, 4, 4};
    assert(y.data().get_shape() == expected_shape);

    std::cout << "[Conv2d Forward Test Passed]" << std::endl;
}

void test_conv2d_backward() {
    using T = float;

    // 입력 데이터 (batch=1, in_channels=1, H=4, W=4)
    Tensor<T> x_data({1, 1, 4, 4}, 1.0f);
    Variable x(x_data);

    // Conv2d 레이어: out_channels=2, kernel_size=3x3, stride=1, pad=1
    layer::Conv2d conv(2, {3, 3}, {1, 1}, {1, 1}, false, 1);

    // Forward
    Variable y = conv.forward({x});
	y.show();

    // Backward
	y.backward();

	y.grad().show();

    // x.grad() shape 검증 (입력과 동일해야 함)
    std::vector<size_t> expected_grad_shape = {1, 1, 4, 4};
    assert(x.grad().data().get_shape() == expected_grad_shape);

    std::cout << "[Conv2d Backward Test Passed]" << std::endl;
}

int main() {
	test_conv2d_forward();
	test_conv2d_backward();
	return 0;
}
