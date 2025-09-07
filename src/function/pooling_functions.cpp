#include "function/pooling_functions.hpp"
#include "container/tensor/tensor_all.hpp"

// [Max Pooling]

Variable function::Pooling::forward(
		const std::vector<Variable>& xs) {
	const Tensor<> &x = xs[0].data();
	input_shape = x.get_shape();

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
	const Variable gx = pooling2d_grad(
								gy, 
								indexes, 
								input_shape, 
								kernel_size, 
								stride, 
								pad);
	return {gx};
}

Variable function::Pooling2DGrad::forward(
		const std::vector<Variable>& xs) {
	const Tensor<>& gy = xs[0].data();

	std::vector<size_t> gy_shape = gy.get_shape();
	size_t N = gy_shape[0];
	size_t C = gy_shape[1];
	size_t OH = gy_shape[2];
	size_t OW = gy_shape[3];

	size_t H = input_shape[2];
	size_t W = input_shape[3];
	auto [KH, KW] = kernel_size;
	size_t K = KH * KW;
	size_t S = N * C * OH * OW;

	Tensor<> gy_flat = gy.ravel();
	Tensor<size_t> idx_flat = indexes.ravel(); 

	Tensor<> gcol_flat = Tensor<>({S * K});

	for (size_t i = 0; i < S; i++) {
		const size_t pos = i * K + idx_flat({i});
		gcol_flat({pos}) = gy_flat({i});
	}
	
	// indexes for python style
	//indexes = indexes.ravel() + arrange<size_t>(0, indexes.size() * KH * KW, KH * KW);
	
	Tensor<> gcol = gcol_flat.reshape({N, C, OH, OW, KH, KW}).transpose({0, 1, 4, 5, 2, 3});

	Tensor<> gx = col2im_array(gcol, {N, C, H, W}, {KH, KW}, stride, pad, false);

	return Variable(gx);
}

std::vector<Variable> function::Pooling2DGrad::backward(
		const Variable& ggy) {
	const Variable ggx = pooling2d_with_indexes(
								ggy, 
								indexes, 
								input_shape, 
								kernel_size, 
								stride, 
								pad);
	return {ggx};
}

Variable function::Pooling2DWithIndexes::forward(
		const std::vector<Variable>& xs) {
	const Tensor<>& x = xs[0].data();
	
	// [N, C, KH, KW, OH, OW]
	Tensor<> col = im2col_array(x, kernel_size, stride, pad, false);

	std::vector<size_t> col_shape = col.get_shape();
	size_t N = col_shape[0];
	size_t C = col_shape[1];
	size_t KH = col_shape[2];
	size_t KW = col_shape[3];
	size_t OH = col_shape[4];
	size_t OW = col_shape[5];
	size_t K = KH * KW;
	size_t S = N * C * OH * OW;

	// [N, C, KH*KW, OH, OW] -> [N, C, OH, OW, KH*KW] -> [S, K]
	col = col.reshape({N, C, K, OH, OW})
				.transpose({0, 1, 3, 4, 2})
				.reshape({S, K});

	// [N, C, OH, OW] -> [S]
	Tensor<size_t> idx_flat = indexes.ravel();

	Tensor<> out_flat({S});
	Tensor<> col_flat = col.ravel();
	for (size_t i = 0; i < S; i++) {
		size_t j = idx_flat({i});
		out_flat({i}) = col_flat({i * K + j});
	}
	
	// [N, C, OH, OW]
	Tensor<> out = out_flat.reshape({N, C, OH, OW});
	return Variable(out);
}

std::vector<Variable> function::Pooling2DWithIndexes::backward(
		const Variable& gy) {
	if (gy.empty()) 
		throw std::runtime_error("Pooling2DWithIndexes backward not implemented");
	return { Variable() };
}

