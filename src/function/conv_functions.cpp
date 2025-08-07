#include "function/conv_functions.hpp"
#include "container/tensor/tensor_all.hpp"

// [Convolution]

Variable function::Conv2d::forward(const std::vector<Variable>& xs) {	
	// [N, C, H, W]
	const Tensor<> &x = xs[0].data();
	// [OC, C, KH, KW]
	const Tensor<> &W = xs[1].data();
	const Tensor<> &b = xs[2].data();
	const auto w_shape = W.get_shape();
	size_t KH = w_shape[2];
	size_t KW = w_shape[3];

	// [N, C, KH, KW, OH, OW]
	const Tensor<> &col = im2col_array(x, {KH, KW}, stride, pad, false);
	
	//col.show();
	//W.show();

	// [N, OH, OW, OC] 
	Tensor<> y = tensordot(col, W, {{1,2,3}, {1,2,3}});
	
	//y.show();

	if (!b.empty())
		y += b;
	
	// [N, OC, OH, OW]
	y = y.transpose({0, 3, 1, 2});

	return Variable(y);
}

std::vector<Variable> function::Conv2d::backward(const Variable& gy) {
	Variable x(inputs[0]);
	Variable W(inputs[1]);
	Variable b(inputs[2]);

	const auto &x_shape = x.shape(); 
	Variable gx = deconv2d(gy, W, b, stride, pad, {x_shape[2], x_shape[3]});

	Variable gW = conv2dgradw(x, gy, W, stride, pad);

	Variable gb;
	if (!b.empty())
		gb = sum(gy, {0,2,3});
	return {gx, gW, gb};
}

Variable function::Deconv2d::forward(const std::vector<Variable>& xs) {	
	// [N, C_in, H_in, W_in]
	const Tensor<> &x = xs[0].data();
	// [C, OC, KH, KW]
	const Tensor<> &W = xs[1].data();
	const Tensor<> &b = xs[2].data();

	auto [SH, SW] = stride;
	auto [PH, PW] = pad;
	
	const auto x_shape = x.get_shape();
	const auto w_shape = W.get_shape();
	
	size_t C = w_shape[0];
	size_t OC = w_shape[1];
	size_t KH = w_shape[2];
	size_t KW = w_shape[3];

	size_t N = x_shape[0];
	size_t C_in = x_shape[1];
	size_t H_in = x_shape[2];
	size_t W_in = x_shape[3];

	if (C != C_in)
		throw std::runtime_error("Deconv2d: W.shape[0] != x.shape[1]");

	size_t OH, OW;
	if (out_size == std::pair<size_t, size_t>{0,0}) {
		OH = get_deconv_outsize(H_in, KH, SH, PH);
		OW = get_deconv_outsize(W_in, KW, SW, PW);
	} else {
		OH = out_size.first;
		OW = out_size.second;
	}
	std::vector<size_t> output_shape = {N, OC, OH, OW};

	// [OC, KH, KW, N, H, W]
	Tensor<> gcol = tensordot(W, x, {{0}, {1}});
	// [N, OC, KH, KW, H, W]
	gcol = gcol.transpose({3, 0, 1, 2, 4, 5});
	
	// [N, OC, OH, OW]
	Tensor<> y = col2im_array(gcol, output_shape, {KH, KW}, stride, pad, false);

	if (!b.empty()) {
		Tensor<> reshaped_b = b.reshape({1, b.size(), 1, 1});
		y += reshaped_b;
	}

	return Variable(y);
		
}

std::vector<Variable> function::Deconv2d::backward(const Variable& gy) {
	Variable x(inputs[0]);
	Variable W(inputs[1]);
	Variable b(inputs[2]);

	Variable gx = conv2d(gy, W, Tensor<>(), stride, pad);
	Variable gW = conv2dgradw(gy, x, W, stride, pad);
	Variable gb;
	if (!b.empty())
		gb = sum(gy, {0, 2, 3});
	return {gx, gW, gb};
}

Variable function::Conv2dGradW::forward(const std::vector<Variable>& xs) {	
	const Tensor<> &x = xs[0].data();
	const Tensor<> &gy = xs[1].data();

	Tensor<> col = im2col_array(x, kernel_size, stride, pad, false);
	Tensor<> gW = tensordot(gy, col, {{0,2,3}, {0,4,5}});

	return Variable(gW);
}

std::vector<Variable> function::Conv2dGradW::backward(const Variable& gW) {
	const Variable &x = inputs[0];
	const Variable &gy = inputs[1];
	const Variable b;

	size_t xh = x.shape()[0];
	size_t xw = x.shape()[1];
	Variable gx = deconv2d(gy, gW, b, stride, pad, {xh, xw});

	Variable ggy = conv2d(x, gW, b, stride, pad);
	return {gx, ggy};
}

// [Im2col]

Variable function::Im2col::forward(const std::vector<Variable>& xs) {
	const Tensor<>& x = xs[0].data();
	input_shape = x.get_shape();
	Tensor<> y = im2col_array(x, kernel_size, stride, pad, to_matrix);
	
	return Variable(y);
}

std::vector<Variable> function::Im2col::backward(const Variable& gy) {
	const Variable gx = col2im(gy, input_shape, kernel_size, stride, pad, to_matrix);
	return {gx};
}


Variable function::Col2im::forward(const std::vector<Variable>& xs) {
	const Tensor<>& x = xs[0].data();
	Tensor<> y = col2im_array(x, input_shape, kernel_size, stride, pad, to_matrix);
	
	return Variable(y);
}

std::vector<Variable> function::Col2im::backward(const Variable& gy) {
	const Variable gx = im2col(gy, kernel_size, stride, pad, to_matrix);
	return {gx};
}


