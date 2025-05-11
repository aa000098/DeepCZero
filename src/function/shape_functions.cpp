#include "function/shape_functions.hpp"

Variable Reshape::forward(const std::vector<Variable>& xs) {
	const Tensor<>& x_data = xs[0].data();
	x_shape = x_data.get_shape();

	Tensor<> result = x_data.reshape(shape);
	return Variable(result);
}

std::vector<Variable> Reshape::backward(const Variable& gy) {
	return { reshape(gy, x_shape) };
}


