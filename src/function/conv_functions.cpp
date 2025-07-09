#include "function/conv_functions.hpp"
#include "container/tensor/tensor_all.hpp"

Variable function::Im2col::forward(const std::vector<Variable>& xs) {
	const Tensor<>& x = xs[0].data();
	std::vector<size_t> x_shape = x.get_shape();
	
	return Variable(x);
}

std::vector<Variable> function::Im2col::backward(const Variable& gy) {
	const Variable& x = inputs[0];
	return {gy * x};
}
