#include "deepczero.hpp"
#include <cassert>
#include <iostream>

void test_mean_squared_error_forward() {
	std::cout << "[Test] MeanSquaredError forward()" << std::endl;

	// x0 = [1, 2, 3], x1 = [2, 2, 2] → diff = [-1, 0, 1]
	Tensor<> t0({3}, {1, 2, 3});
	Tensor<> t1({3}, {2, 2, 2});

	Variable x0(t0);
	Variable x1(t1);

	Variable loss = mean_squared_error(x0, x1);

	float expected = (1.0f + 0.0f + 1.0f) / 3.0f;  // = 0.666...
	assert(std::abs(loss.data()({0}) - expected) < 1e-5);
}

void test_mean_squared_error_backward() {
	std::cout << "[Test] MeanSquaredError backward()" << std::endl;

	Tensor<> t0({3}, {1, 2, 3});
	Tensor<> t1({3}, {2, 2, 2});

	Variable x0(t0);
	Variable x1(t1);

	Variable loss = mean_squared_error(x0, x1);  // wrapper function or use Function directly
	loss.backward();

	// expected gradient: dL/dx0 = (2/N) * (x0 - x1)
	std::vector<float> expected_grad = {-0.6666f, 0.0f, 0.6666f};

	for (size_t i = 0; i < 3; ++i) {
		float g = x0.grad().data()({i});
		assert(std::abs(g - expected_grad[i]) < 1e-3);
	}
}

int main() {
	test_mean_squared_error_forward();
	test_mean_squared_error_backward();

	std::cout << "✅ All MeanSquaredError tests passed." << std::endl;
	return 0;
}

