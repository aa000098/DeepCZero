#include "deepczero.hpp"

#include <cassert>
#include <iostream>

void test_cbs_forward() {
	// Input: batch=1, channels=3, H=8, W=8
	Tensor<> x_data = randn({1, 3, 8, 8}, 42);
	Variable x(x_data);

	// CBS with 3x3 kernel, stride=1, pad=1 -> same spatial size
	layer::CBS cbs(3, 16, {3, 3}, {1, 1}, {1, 1});
	Variable y = cbs(x);

	assert(y.shape() == std::vector<size_t>({1, 16, 8, 8}));
	std::cout << "[CBS Forward] Passed - output shape: [1, 16, 8, 8]" << std::endl;
}

void test_cbs_stride2() {
	// CBS with stride=2 -> spatial size halved
	Tensor<> x_data = randn({1, 16, 8, 8}, 42);
	Variable x(x_data);

	layer::CBS cbs(16, 32, {3, 3}, {2, 2}, {1, 1});
	Variable y = cbs(x);

	assert(y.shape() == std::vector<size_t>({1, 32, 4, 4}));
	std::cout << "[CBS Stride=2] Passed - output shape: [1, 32, 4, 4]" << std::endl;
}

void test_bottleneck_forward() {
	Tensor<> x_data = randn({1, 16, 8, 8}, 42);
	Variable x(x_data);

	// Shortcut enabled: in_channels == out_channels
	layer::Bottleneck bn(16, 16, true);
	Variable y = bn(x);

	assert(y.shape() == std::vector<size_t>({1, 16, 8, 8}));
	std::cout << "[Bottleneck Forward (shortcut)] Passed - output shape: [1, 16, 8, 8]" << std::endl;
}

void test_bottleneck_no_shortcut() {
	Tensor<> x_data = randn({1, 16, 8, 8}, 42);
	Variable x(x_data);

	// Shortcut disabled or mismatched channels
	layer::Bottleneck bn(16, 32, true);  // shortcut disabled due to channel mismatch
	Variable y = bn(x);

	assert(y.shape() == std::vector<size_t>({1, 32, 8, 8}));
	std::cout << "[Bottleneck Forward (no shortcut)] Passed - output shape: [1, 32, 8, 8]" << std::endl;
}

void test_c3_forward() {
	Tensor<> x_data = randn({1, 16, 8, 8}, 42);
	Variable x(x_data);

	layer::C3 c3(16, 16, /*n=*/1);
	Variable y = c3(x);

	assert(y.shape() == std::vector<size_t>({1, 16, 8, 8}));
	std::cout << "[C3 Forward] Passed - output shape: [1, 16, 8, 8]" << std::endl;
}

void test_c3_multiple_bottlenecks() {
	Tensor<> x_data = randn({1, 32, 8, 8}, 42);
	Variable x(x_data);

	layer::C3 c3(32, 32, /*n=*/3);
	Variable y = c3(x);

	assert(y.shape() == std::vector<size_t>({1, 32, 8, 8}));
	std::cout << "[C3 Forward (n=3)] Passed - output shape: [1, 32, 8, 8]" << std::endl;
}

void test_sppf_forward() {
	Tensor<> x_data = randn({1, 32, 8, 8}, 42);
	Variable x(x_data);

	layer::SPPF sppf(32, 32, /*k=*/5);
	Variable y = sppf(x);

	assert(y.shape() == std::vector<size_t>({1, 32, 8, 8}));
	std::cout << "[SPPF Forward] Passed - output shape: [1, 32, 8, 8]" << std::endl;
}

int main() {
	test_cbs_forward();
	test_cbs_stride2();
	test_bottleneck_forward();
	test_bottleneck_no_shortcut();
	test_c3_forward();
	test_c3_multiple_bottlenecks();
	test_sppf_forward();

	std::cout << "\nAll YOLOv5 module tests passed!" << std::endl;
	return 0;
}
