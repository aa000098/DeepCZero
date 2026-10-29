#include "function/pooling_functions.hpp"
#include "container/tensor/tensor_all.hpp"
#include "config/config.hpp"

#include <limits>

// [Max Pooling]

Variable function::Pooling::forward(
		const std::vector<Variable>& xs) {
	const Tensor<> &x_orig = xs[0].data();
	input_shape = x_orig.get_shape();

#ifdef USE_CUDA
	if (x_orig.device().type == dcz::DeviceType::CUDA) {
		auto [y, idx] = tensor::maxpool_forward_cuda(x_orig,
			kernel_size.first, kernel_size.second,
			stride.first, stride.second,
			pad.first, pad.second,
			dcz::Config::get().enable_backprop);
		indexes = idx;
		return Variable(y);
	}
#endif

	Tensor<> x = (x_orig.is_device() ? x_orig.cpu() : x_orig).contiguous();

	size_t N = input_shape[0];
	size_t C = input_shape[1];
	size_t H = input_shape[2];
	size_t W = input_shape[3];
	auto [KH, KW] = kernel_size;
	auto [SH, SW] = stride;
	auto [PH, PW] = pad;
	size_t OH = get_conv_outsize(H, KH, SH, PH);
	size_t OW = get_conv_outsize(W, KW, SW, PW);

	const bool need_idx = dcz::Config::get().enable_backprop;

	Tensor<> y({N, C, OH, OW});
	if (need_idx)
		indexes = Tensor<size_t>({N, C, OH, OW}, 0);
	else
		indexes = Tensor<size_t>();

	const float* xp = x.raw_data().data();
	float* yp = y.raw_data().data();
	size_t* ip = need_idx ? indexes.raw_data().data() : nullptr;

	// indexes[o] = kh * KW + kw of the max inside the window
	#pragma omp parallel for collapse(2)
	for (size_t n = 0; n < N; ++n) {
		for (size_t c = 0; c < C; ++c) {
			const float* xc = xp + (n * C + c) * H * W;
			float* yc = yp + (n * C + c) * OH * OW;
			size_t* ic = ip ? ip + (n * C + c) * OH * OW : nullptr;

			for (size_t oh = 0; oh < OH; ++oh) {
				for (size_t ow = 0; ow < OW; ++ow) {
					float max_val = -std::numeric_limits<float>::infinity();
					size_t max_idx = 0;
					for (size_t kh = 0; kh < KH; ++kh) {
						const size_t h_in = oh * SH + kh;
						if (h_in < PH || h_in >= H + PH) continue;
						const float* xrow = xc + (h_in - PH) * W;
						for (size_t kw = 0; kw < KW; ++kw) {
							const size_t w_in = ow * SW + kw;
							if (w_in < PW || w_in >= W + PW) continue;
							const float v = xrow[w_in - PW];
							if (v > max_val) {
								max_val = v;
								max_idx = kh * KW + kw;
							}
						}
					}
					yc[oh * OW + ow] = max_val;
					if (ic) ic[oh * OW + ow] = max_idx;
				}
			}
		}
	}

	if (x_orig.is_device())
		y = y.to(x_orig.device());
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
	const Tensor<>& gy_orig = xs[0].data();
	Tensor<> gy = (gy_orig.is_device() ? gy_orig.cpu() : gy_orig).contiguous();
	Tensor<size_t> idx = indexes.contiguous();

	size_t N = input_shape[0];
	size_t C = input_shape[1];
	size_t H = input_shape[2];
	size_t W = input_shape[3];
	auto [KH, KW] = kernel_size;
	auto [SH, SW] = stride;
	auto [PH, PW] = pad;
	size_t OH = get_conv_outsize(H, KH, SH, PH);
	size_t OW = get_conv_outsize(W, KW, SW, PW);

	Tensor<> gx({N, C, H, W}, 0.0f);
	const float* gyp = gy.raw_data().data();
	const size_t* ip = idx.raw_data().data();
	float* gxp = gx.raw_data().data();

	// Windows inside one (n, c) plane may overlap: scatter per plane,
	// parallelise over planes only.
	#pragma omp parallel for collapse(2)
	for (size_t n = 0; n < N; ++n) {
		for (size_t c = 0; c < C; ++c) {
			const float* gyc = gyp + (n * C + c) * OH * OW;
			const size_t* ic = ip + (n * C + c) * OH * OW;
			float* gxc = gxp + (n * C + c) * H * W;

			for (size_t oh = 0; oh < OH; ++oh) {
				for (size_t ow = 0; ow < OW; ++ow) {
					const size_t o = oh * OW + ow;
					const size_t h_in = oh * SH + ic[o] / KW;
					const size_t w_in = ow * SW + ic[o] % KW;
					if (h_in < PH || h_in >= H + PH) continue;
					if (w_in < PW || w_in >= W + PW) continue;
					gxc[(h_in - PH) * W + (w_in - PW)] += gyc[o];
				}
			}
		}
	}

	if (gy_orig.is_device())
		gx = gx.to(gy_orig.device());

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
	const Tensor<>& x_orig = xs[0].data();
	Tensor<> x = (x_orig.is_device() ? x_orig.cpu() : x_orig).contiguous();
	Tensor<size_t> idx = indexes.contiguous();

	std::vector<size_t> x_shape = x.get_shape();
	size_t N = x_shape[0];
	size_t C = x_shape[1];
	size_t H = x_shape[2];
	size_t W = x_shape[3];
	auto [KH, KW] = kernel_size;
	auto [SH, SW] = stride;
	auto [PH, PW] = pad;
	size_t OH = get_conv_outsize(H, KH, SH, PH);
	size_t OW = get_conv_outsize(W, KW, SW, PW);

	Tensor<> out({N, C, OH, OW}, 0.0f);
	const float* xp = x.raw_data().data();
	const size_t* ip = idx.raw_data().data();
	float* op = out.raw_data().data();

	#pragma omp parallel for collapse(2)
	for (size_t n = 0; n < N; ++n) {
		for (size_t c = 0; c < C; ++c) {
			const float* xc = xp + (n * C + c) * H * W;
			const size_t* ic = ip + (n * C + c) * OH * OW;
			float* oc = op + (n * C + c) * OH * OW;

			for (size_t oh = 0; oh < OH; ++oh) {
				for (size_t ow = 0; ow < OW; ++ow) {
					const size_t o = oh * OW + ow;
					const size_t h_in = oh * SH + ic[o] / KW;
					const size_t w_in = ow * SW + ic[o] % KW;
					if (h_in < PH || h_in >= H + PH) continue;
					if (w_in < PW || w_in >= W + PW) continue;
					oc[o] = xc[(h_in - PH) * W + (w_in - PW)];
				}
			}
		}
	}

	if (x_orig.is_device())
		out = out.to(x_orig.device());
	return Variable(out);
}

std::vector<Variable> function::Pooling2DWithIndexes::backward(
		const Variable& gy) {
	if (gy.empty())
		throw std::runtime_error("Pooling2DWithIndexes backward not implemented");
	return { Variable() };
}

