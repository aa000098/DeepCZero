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


Variable Transpose::forward(const std::vector<Variable>& xs) {
    const Tensor<>& x_data = xs[0].data();
    Tensor<> result = x_data.transpose(axes);
    return Variable(result);
}

std::vector<Variable> Transpose::backward(const Variable& gy) {
	if (axes.empty())
		return {transpose(gy)};

	size_t ndim = axes.size();
	std::vector<size_t> inv_axes(ndim); 
	for (size_t i = 0; i < ndim; i++)
		inv_axes[axes[i]] = i;
    return { transpose(gy, inv_axes) };
}

