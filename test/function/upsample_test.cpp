#include "deepczero.hpp"

#include <iostream>
#include <cassert>

void test_upsample_forward() {
	std::cout << "=== Upsample Forward ===" << std::endl;

	// [1, 1, 2, 2] -> [1, 1, 4, 4] with scale=2
	Tensor<> x_data({1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
	Variable x(x_data);

	Variable y = upsample(x, 2);

	auto shape = y.shape();
	std::cout << "Input shape: [1,1,2,2] -> Output shape: ["
			  << shape[0] << "," << shape[1] << ","
			  << shape[2] << "," << shape[3] << "]" << std::endl;

	assert(shape[0] == 1);
	assert(shape[1] == 1);
	assert(shape[2] == 4);
	assert(shape[3] == 4);

	// Each pixel is repeated 2x2
	assert(y.data()({0, 0, 0, 0}) == 1.0f);
	assert(y.data()({0, 0, 0, 1}) == 1.0f);
	assert(y.data()({0, 0, 1, 0}) == 1.0f);
	assert(y.data()({0, 0, 1, 1}) == 1.0f);
	assert(y.data()({0, 0, 0, 2}) == 2.0f);
	assert(y.data()({0, 0, 2, 0}) == 3.0f);
	assert(y.data()({0, 0, 2, 2}) == 4.0f);

	std::cout << "Upsample forward test passed!" << std::endl;
}

void test_upsample_backward() {
	std::cout << "=== Upsample Backward ===" << std::endl;

	Tensor<> x_data({1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
	Variable x(x_data);

	Variable y = upsample(x, 2);
	Variable loss = sum(y);
	loss.backward();

	// Each input pixel contributes to scale^2 = 4 output pixels
	// So gradient should be 4.0 for each input pixel
	assert(x.has_grad());
	assert(x.grad().data()({0, 0, 0, 0}) == 4.0f);
	assert(x.grad().data()({0, 0, 1, 1}) == 4.0f);

	std::cout << "Upsample backward test passed!" << std::endl;
}

int main() {
	test_upsample_forward();
	test_upsample_backward();
	std::cout << "\nAll Upsample tests passed!" << std::endl;
	return 0;
}
