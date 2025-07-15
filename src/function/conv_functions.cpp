#include "function/conv_functions.hpp"
#include "container/tensor/tensor_all.hpp"

Variable function::Im2col::forward(const std::vector<Variable>& xs) {
	const Tensor<>& x = xs[0].data();
	input_shape = x.get_shape();
	Tensor<> y = im2col_array(x, kernel_size, stride, pad, to_matrix);
	
	return Variable(y);
}

std::vector<Variable> function::Im2col::backward(const Variable& gy) {
	const Variable& x = inputs[0];
	return {gy * x};
}


