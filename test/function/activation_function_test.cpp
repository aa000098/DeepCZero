#include <cassert>
#include <cmath>
#include <iostream>
#include "deepczero.hpp"  // Include this according to your framework setup

#define DCZ_TEST_MODE() auto __dcz_config_guard = dcz::test_mode()

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

void test_softmax_forward_backward() {
    std::cout << "▶️ Running Softmax forward/backward test...\n";

	using T = float;


	// 입력: 2x3 행렬
	Tensor<T> x_data({2, 3}, {
		1.0, 2.0, 3.0,
		0.0, 0.0, 0.0
	});
	Variable x(x_data);

	// softmax(axis=1)
	Variable y = softmax(x, {1});

	// expected 결과: 행마다 softmax 수행
	std::vector<T> expected_y(6);
	{
		// 첫 번째 행: [1.0, 2.0, 3.0]
		float sum1 = std::exp(1.0f) + std::exp(2.0f) + std::exp(3.0f);
		expected_y[0] = std::exp(1.0f) / sum1;
		expected_y[1] = std::exp(2.0f) / sum1;
		expected_y[2] = std::exp(3.0f) / sum1;

		// 두 번째 행: [0.0, 0.0, 0.0] → softmax = uniform
		expected_y[3] = expected_y[4] = expected_y[5] = 1.0f / 3.0f;
	}

	const auto& y_data = y.data().raw_data();
	for (size_t i = 0; i < y_data.size(); ++i) {
		assert(std::abs(y_data[i] - expected_y[i]) < 1e-5);
	}

	// 역전파 테스트
	y.backward();

	// 역전파 결과 확인 (일단 합이 0인지 확인: softmax의 특성)
	const auto& gx_data = x.grad().data().raw_data();
	for (size_t row = 0; row < 2; ++row) {
		float sum = gx_data[row * 3 + 0] + gx_data[row * 3 + 1] + gx_data[row * 3 + 2];
		assert(std::abs(sum) < 1e-5);
	}

	std::cout << "✅ Softmax forward/backward test passed.\n";

}

void test_relu_forward_backward() {
    using namespace function;

    // 입력 데이터 준비
    Tensor<float> input_tensor({6}, {-1.0f, 0.0f, 1.0f, 2.0f, -0.5f, 3.5f});
    Variable x(input_tensor);

    // 순전파 테스트
    Variable y = relu(x);
    Tensor<float> expected_forward({6}, {0.0f, 0.0f, 1.0f, 2.0f, 0.0f, 3.5f});
    for (size_t i = 0; i < y.data().size(); ++i) {
        assert(std::abs(y.data().data()[i] - expected_forward.data()[i]) < 1e-6f);
    }

    // 역전파 테스트: dy는 모두 1이라고 가정

    y.backward();
	Tensor<float> grads = x.grad().data();
    Tensor<float> expected_grad({6}, {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f});
    for (size_t i = 0; i < grads.data().size(); ++i) {
        assert(std::abs(grads.data()[i] - expected_grad.data()[i]) < 1e-6f);
    }

    std::cout << "ReLU forward and backward test passed!" << std::endl;
}

void test_dropout_forward() {
    using namespace function;

    // (1) 입력 데이터 준비
    Tensor<float> input_tensor({6}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Variable x(input_tensor);

    // (2) 학습 모드 테스트 (Config::train = true)
    Variable y_train = dropout(x, 0.5f);

    std::cout << "[Train mode] Dropout result: ";
    for (size_t i = 0; i < y_train.data().size(); ++i)
        std::cout << y_train.data().data()[i] << " ";
    std::cout << std::endl;

    // (3) 평가 모드 테스트 (Config::train = false)
	{
		DCZ_TEST_MODE();
    	Variable y_eval = dropout(x, 0.5f);
    	std::cout << "[Eval mode] Dropout result: ";
    	for (size_t i = 0; i < y_eval.data().size(); ++i)
        	std::cout << y_eval.data().data()[i] << " ";
    	std::cout << std::endl;

    // (4) assert로 평가 모드 검증: 입력과 동일해야 함
    	for (size_t i = 0; i < y_eval.data().size(); ++i) {
        	assert(std::abs(y_eval.data().data()[i] - input_tensor.data()[i]) < 1e-6f);
    	}
	}

    std::cout << "✅ Dropout forward test passed (train + eval mode)!" << std::endl;
}

int main() {
    test_sigmoid_forward_backward();
	test_softmax_forward_backward();
	test_relu_forward_backward();
	test_dropout_forward();
    return 0;
}

