#include "function/activation_functions.hpp"
#include "container/tensor/tensor_all.hpp"

Variable function::Sigmoid::forward(const std::vector<Variable>& xs) {
	const Tensor<>& x = xs[0].data();
	const Tensor<> result = tanh(x * 0.5f) * 0.5f + 0.5f;
	return Variable(result);
}

std::vector<Variable> function::Sigmoid::backward(const Variable& gy) {
	Variable y = output.lock();

	Variable gx = gy * y * (1.0f - y);
	return { gx };
}
