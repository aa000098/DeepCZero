#include <cassert>
#include <cmath>
#include <iostream>
#include "deepczero.hpp"  // Include this according to your framework setup

float sigmoid_reference(float x) {
    return 0.5f * std::tanh(0.5f * x) + 0.5f;
}

float sigmoid_grad_reference(float y) {
    return y * (1.0f - y);
}

void test_sigmoid_forward_backward() {
    std::cout << "▶️ Running Sigmoid forward/backward test...\n";

    // 1. 입력값 설정
    Tensor<> x_tensor({3}, {-2.0f, 0.0f, 2.0f});
    Variable x(x_tensor);

    // 2. Forward
    Variable y = sigmoid(x);

    // 3. Forward 결과 검증
    const auto& y_data = y.data().raw_data();
    for (size_t i = 0; i < y_data.size(); ++i) {
        float expected = sigmoid_reference(x_tensor.raw_data()[i]);
        std::cout << "  y[" << i << "] = " << y_data[i] << ", expected = " << expected << std::endl;
        assert(std::abs(y_data[i] - expected) < 1e-5f);
    }

    // 4. Backward (gy = [1, 1, 1])
    //Tensor<> gy_tensor({3}, {1.0f, 1.0f, 1.0f});
    //Variable gy(gy_tensor);
    //y.set_grad(gy);
    y.backward();

    // 5. Backward 결과 검증
    const auto& gx_data = x.grad().data().raw_data();
    for (size_t i = 0; i < gx_data.size(); ++i) {
        float expected_grad = sigmoid_grad_reference(y_data[i]);  // dy/dx = y * (1 - y)
        std::cout << "  gx[" << i << "] = " << gx_data[i] << ", expected = " << expected_grad << std::endl;
        assert(std::abs(gx_data[i] - expected_grad) < 1e-5f);
    }

    std::cout << "✅ Sigmoid forward/backward test passed.\n";
}


int main() {
    test_sigmoid_forward_backward();
    return 0;
}

