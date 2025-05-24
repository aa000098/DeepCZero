#include "deepczero.hpp"

#include "deepczero.hpp"
#include <cassert>
#include <iostream>

void test_sum_backward_scalar() {
	std::cout << "[Test] sum backward (scalar)" << std::endl;
	Tensor<> t({3, 1}, {1, 2, 3});
	Variable x(t);
	Variable y = sum(x);  // 전체 sum, 결과는 scalar
	x.set_name("x");
	y.set_name("y");

	x.show();
	y.show();
	std::cout << "x.shape(): "; 
	for (auto s : x.shape()) 
		std::cout << s << " ";
	std::cout << std::endl;
	std::cout << "y.shape(): "; 
	for (auto s : y.shape()) 
		std::cout << s << " ";
	std::cout << std::endl;

	y.backward();

	assert(x.grad().shape() == x.shape());

	for (size_t i = 0; i < 3; ++i) {
		float g = x.grad().data()({i, 0});
		assert(std::abs(g - 1.0f) < 1e-5);
	}
}

void test_sum_backward_axis0() {
	std::cout << "[Test] sum backward (axis=0)" << std::endl;
	Tensor<> t({2, 3}, {
		1, 2, 3,
		4, 5, 6
	});
	Variable x(t);
	Variable y = sum(x, {0});  // shape = [3]

	y.backward();

	assert(x.grad().shape() == x.shape());

	for (size_t i = 0; i < 2; ++i) {
		for (size_t j = 0; j < 3; ++j) {
			float g = x.grad().data()({i, j});
			assert(std::abs(g - 1.0f) < 1e-5);
		}
	}
}

void test_sum_backward_keepdims() {
	std::cout << "[Test] sum backward (axis=1, keepdims=true)" << std::endl;
	Tensor<> t({2, 3}, {
		1, 2, 3,
		4, 5, 6
	});
	Variable x(t);
	Variable y = sum(x, {1}, true);  // shape = [2, 1]

	y.backward();

	assert(x.grad().shape() == x.shape());

	for (size_t i = 0; i < 2; ++i) {
		for (size_t j = 0; j < 3; ++j) {
			float g = x.grad().data()({i, j});
			assert(std::abs(g - 1.0f) < 1e-5);
		}
	}
}

int main() {
	test_sum_backward_scalar();
	test_sum_backward_axis0();
	test_sum_backward_keepdims();

	std::cout << "✅ All sum backward tests passed." << std::endl;
	return 0;
}

