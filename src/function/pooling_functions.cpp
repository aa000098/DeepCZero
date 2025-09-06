#include "function/pooling_functions.hpp"
#include "container/tensor/tensor_all.hpp"

// [Max Pooling]

Variable function::Pooling::forward(
		const std::vector<Variable>& xs) {
	const Tensor<> &x = xs[0].data();

	Tensor<> col = im2col_array(x, kernel_size, stride, pad, false);

	std::vector<size_t> col_shape = col.get_shape();
	
	size_t N = col_shape[0];
	size_t C = col_shape[1];
	size_t KH = col_shape[2];
	size_t KW = col_shape[3];
	size_t OH = col_shape[4];
	size_t OW = col_shape[5];

	col = col.reshape({N, C, KH * KW, OH, OW});

	indexes = col.argmax(2);
	Tensor<> y = col.max({2});
	return Variable(y);
}

std::vector<Variable> function::Pooling::backward(
		const Variable& gy) {

	return { Variable() };
}
