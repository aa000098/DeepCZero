#include "ops/ops.hpp"
#include "container/variable.hpp"
#include "function/function.hpp"

#include <iostream>

using Tensor = std::vector<float>;

int main() {
	Variable x({0.5});

	Variable y1 = square(x);
	Variable y2 = exp(y1);
	Variable y3 = square(y2);
	y3.show();

	Tensor x2({1});
	std::shared_ptr<Function> C = y3.get_creator();
	std::vector<Tensor> grad_C = C->backward(x2);
	std::cout << "[grad_C]: " << grad_C[0][0] << std::endl;

	std::shared_ptr<Function> B = y2.get_creator();
	std::vector<Tensor> grad_B = B->backward(grad_C[0]);
	std::cout << "[grad_B]: " << grad_B[0][0] << std::endl;

	std::shared_ptr<Function> A = y1.get_creator();
	std::vector<Tensor> grad_A = A->backward(grad_B[0]);
	std::cout << "[grad_A]: " << grad_A[0][0] << std::endl;
}
