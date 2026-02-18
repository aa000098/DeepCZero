#include "deepczero.hpp"

#include <iostream>
#include <cassert>

void test_concat_forward() {
	std::cout << "=== Concat Forward ===" << std::endl;

	Tensor<> a_data({2, 3, 2, 2}, std::vector<float>(2*3*2*2, 1.0f));
	Tensor<> b_data({2, 5, 2, 2}, std::vector<float>(2*5*2*2, 2.0f));

	Variable a(a_data);
	Variable b(b_data);

	Variable y = concat({a, b}, 1);

	auto shape = y.shape();
	std::cout << "a shape: [2,3,2,2], b shape: [2,5,2,2]" << std::endl;
	std::cout << "concat shape: [" << shape[0] << "," << shape[1] << ","
			  << shape[2] << "," << shape[3] << "]" << std::endl;

	assert(shape[0] == 2);
	assert(shape[1] == 8);  // 3 + 5
	assert(shape[2] == 2);
	assert(shape[3] == 2);

	// Check values
	assert(y.data()({0, 0, 0, 0}) == 1.0f);  // from a
	assert(y.data()({0, 3, 0, 0}) == 2.0f);  // from b

	std::cout << "Concat forward test passed!" << std::endl;
}

void test_concat_backward() {
	std::cout << "=== Concat Backward ===" << std::endl;

	Tensor<> a_data({1, 2, 2, 2}, std::vector<float>(1*2*2*2, 1.0f));
	Tensor<> b_data({1, 3, 2, 2}, std::vector<float>(1*3*2*2, 2.0f));

	Variable a(a_data);
	Variable b(b_data);

	Variable y = concat({a, b}, 1);
	Variable loss = sum(y);
	loss.backward();

	// Gradient should be all ones for both
	assert(a.has_grad());
	assert(b.has_grad());

	auto a_grad_shape = a.grad().shape();
	auto b_grad_shape = b.grad().shape();

	assert(a_grad_shape[0] == 1 && a_grad_shape[1] == 2);
	assert(b_grad_shape[0] == 1 && b_grad_shape[1] == 3);

	std::cout << "Concat backward test passed!" << std::endl;
}

int main() {
	test_concat_forward();
	test_concat_backward();
	std::cout << "\nAll Concat tests passed!" << std::endl;
	return 0;
}
