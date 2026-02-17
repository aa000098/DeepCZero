#include "deepczero.hpp"

#include <iostream>
#include <cmath>

void test_silu_forward() {
	std::cout << "=== SiLU Forward ===" << std::endl;

	Tensor<> x_data({2, 3}, {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f});
	Variable x(x_data);

	Variable y = silu(x);

	std::cout << "Input: ";
	x.show();
	std::cout << "Output: ";
	y.show();

	// SiLU(0) = 0, SiLU(x) = x * sigmoid(x)
	// SiLU(-2) = -2 * sigmoid(-2) = -2 * 0.1192 = -0.2384
	// SiLU(1) = 1 * sigmoid(1) = 0.7311
	const auto& out = y.data();
	assert(std::abs(out({0, 2}) - 0.0f) < 1e-5);
	assert(std::abs(out({0, 0}) - (-2.0f * (1.0f / (1.0f + std::exp(2.0f))))) < 1e-4);
	assert(std::abs(out({1, 0}) - (1.0f / (1.0f + std::exp(-1.0f)))) < 1e-4);

	std::cout << "SiLU forward test passed!" << std::endl;
}

void test_silu_backward() {
	std::cout << "=== SiLU Backward ===" << std::endl;

	Tensor<> x_data({2, 2}, {-1.0f, 0.0f, 1.0f, 2.0f});
	Variable x(x_data);

	Variable y = silu(x);
	y.backward();

	std::cout << "Input: ";
	x.show();
	std::cout << "Output: ";
	y.show();
	std::cout << "Gradient: ";
	x.grad().show();

	// d/dx[x * sigmoid(x)] = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
	// At x=0: sigmoid(0)=0.5, grad = 0.5 + 0*0.5*0.5 = 0.5
	assert(std::abs(x.grad().data()({0, 1}) - 0.5f) < 1e-4);

	std::cout << "SiLU backward test passed!" << std::endl;
}

int main() {
	test_silu_forward();
	test_silu_backward();
	std::cout << "\nAll SiLU tests passed!" << std::endl;
	return 0;
}
