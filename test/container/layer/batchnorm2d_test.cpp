#include "deepczero.hpp"

#include <cassert>
#include <iostream>
#include <cmath>

#define DCZ_TEST_MODE() auto __dcz_config_guard = dcz::test_mode()

void test_batchnorm2d_forward_shape() {
	Tensor<> x_data = randn({4, 8, 16, 16}, 42);
	Variable x(x_data);

	layer::BatchNorm2d bn(8);
	Variable y = bn(x);

	std::vector<size_t> expected = {4, 8, 16, 16};
	assert(y.shape() == expected);
	std::cout << "[BatchNorm2d Forward Shape] Passed" << std::endl;
}

void test_batchnorm2d_constant_input() {
	// With constant input, (x - mean) = 0, so output = beta = 0
	Tensor<> x_data({2, 3, 4, 4}, 5.0f);
	Variable x(x_data);

	layer::BatchNorm2d bn(3);
	Variable y = bn(x);

	assert(y.shape() == std::vector<size_t>({2, 3, 4, 4}));
	for (size_t i = 0; i < y.data().size(); ++i)
		assert(std::abs(y.data().raw_data()[i]) < 1e-5f);

	std::cout << "[BatchNorm2d Constant Input] Passed" << std::endl;
}

void test_batchnorm2d_backward() {
	Tensor<> x_data = randn({2, 3, 4, 4}, 42);
	Variable x(x_data);

	layer::BatchNorm2d bn(3);
	Variable y = bn(x);

	Variable loss = sum(y);
	loss.backward();

	// x should have gradients
	assert(x.has_grad());
	assert(x.grad().shape() == std::vector<size_t>({2, 3, 4, 4}));

	// gamma (weight) should have gradients
	Parameter weight = bn.get_param("weight");
	assert(weight.has_grad());
	assert(weight.grad().shape() == std::vector<size_t>({3}));

	// beta (bias) should have gradients
	Parameter bias = bn.get_param("bias");
	assert(bias.has_grad());
	assert(bias.grad().shape() == std::vector<size_t>({3}));

	std::cout << "[BatchNorm2d Backward] Passed" << std::endl;
}

void test_batchnorm2d_eval_mode() {
	layer::BatchNorm2d bn(3);

	// Do one training forward to populate running stats
	Tensor<> x_data = randn({2, 3, 4, 4}, 42);
	Variable x(x_data);
	Variable y_train = bn(x);

	// Switch to eval mode
	{
		DCZ_TEST_MODE();
		Variable y_eval = bn(x);
		assert(y_eval.shape() == std::vector<size_t>({2, 3, 4, 4}));
		std::cout << "[BatchNorm2d Eval Mode] Passed" << std::endl;
	}
}

void test_batchnorm2d_running_stats_update() {
	layer::BatchNorm2d bn(3);

	// Running mean should start at 0
	for (size_t c = 0; c < 3; ++c)
		assert(std::abs(bn.get_running_mean().raw_data()[c]) < 1e-6f);

	// Create input with known per-channel values
	Tensor<> x_data({2, 3, 4, 4}, 0.0f);
	for (size_t n = 0; n < 2; ++n)
		for (size_t c = 0; c < 3; ++c)
			for (size_t h = 0; h < 4; ++h)
				for (size_t w = 0; w < 4; ++w)
					x_data({n, c, h, w}) = static_cast<float>(c + 1);

	Variable x(x_data);
	Variable y = bn(x);

	// running_mean = (1 - 0.1) * 0 + 0.1 * batch_mean = 0.1 * [1, 2, 3]
	for (size_t c = 0; c < 3; ++c) {
		float expected = 0.1f * static_cast<float>(c + 1);
		float actual = bn.get_running_mean().raw_data()[c];
		assert(std::abs(actual - expected) < 1e-5f);
	}

	std::cout << "[BatchNorm2d Running Stats Update] Passed" << std::endl;
}

void test_batchnorm2d_normalization_values() {
	// With gamma=1, beta=0, output should have mean~0, var~1 per channel
	Tensor<> x_data = randn({8, 2, 4, 4}, 123);
	Variable x(x_data);

	layer::BatchNorm2d bn(2);
	Variable y = bn(x);

	const Tensor<>& y_data = y.data();
	for (size_t c = 0; c < 2; ++c) {
		float channel_sum = 0.0f;
		float channel_sq_sum = 0.0f;
		size_t count = 0;
		for (size_t n = 0; n < 8; ++n) {
			for (size_t h = 0; h < 4; ++h) {
				for (size_t w = 0; w < 4; ++w) {
					float val = y_data({n, c, h, w});
					channel_sum += val;
					channel_sq_sum += val * val;
					count++;
				}
			}
		}
		float mean = channel_sum / count;
		float var = channel_sq_sum / count - mean * mean;

		assert(std::abs(mean) < 1e-4f);
		assert(std::abs(var - 1.0f) < 1e-3f);
	}

	std::cout << "[BatchNorm2d Normalization Values] Passed" << std::endl;
}

int main() {
	test_batchnorm2d_forward_shape();
	test_batchnorm2d_constant_input();
	test_batchnorm2d_backward();
	test_batchnorm2d_eval_mode();
	test_batchnorm2d_running_stats_update();
	test_batchnorm2d_normalization_values();

	std::cout << "\nAll BatchNorm2d tests passed!" << std::endl;
	return 0;
}
