#include "function/shape_functions.hpp"
#include "container/tensor/tensor_all.hpp"


Variable function::Concat::forward(const std::vector<Variable>& xs) {
	size_t n = xs.size();
	if (n == 0)
		throw std::runtime_error("Concat: empty input list");

	const auto& base_shape = xs[0].data().get_shape();
	size_t ndim = base_shape.size();
	int ax = axis;
	if (ax < 0) ax += static_cast<int>(ndim);

	split_sizes.clear();
	split_sizes.reserve(n);

	std::vector<Tensor<>> tensors;
	tensors.reserve(n);
	for (size_t i = 0; i < n; i++) {
		tensors.push_back(xs[i].data());
		split_sizes.push_back(xs[i].data().get_shape()[ax]);
	}

	// Compute output shape
	std::vector<size_t> out_shape = base_shape;
	size_t total_axis_size = 0;
	for (size_t i = 0; i < n; i++) {
		const auto& s = tensors[i].get_shape();
		for (size_t d = 0; d < ndim; d++) {
			if (static_cast<int>(d) != ax && s[d] != base_shape[d])
				throw std::runtime_error("Concat: shape mismatch at dim " + std::to_string(d));
		}
		total_axis_size += s[ax];
	}
	out_shape[ax] = total_axis_size;

	// Compute total size and allocate
	size_t total_size = 1;
	for (auto s : out_shape) total_size *= s;
	std::vector<float> out_data(total_size);

	// Copy data: iterate over outer dims, concat along axis, iterate over inner dims
	size_t outer_size = 1;
	for (int d = 0; d < ax; d++) outer_size *= out_shape[d];
	size_t inner_size = 1;
	for (size_t d = ax + 1; d < ndim; d++) inner_size *= out_shape[d];

	size_t out_offset = 0;
	for (size_t o = 0; o < outer_size; o++) {
		for (size_t i = 0; i < n; i++) {
			const auto& t = tensors[i];
			size_t axis_len = split_sizes[i];
			size_t chunk = axis_len * inner_size;
			const float* src = t.raw_data().data() + o * chunk;
			std::copy(src, src + chunk, out_data.data() + out_offset);
			out_offset += chunk;
		}
	}

	return Variable(Tensor<>(out_shape, out_data));
}

std::vector<Variable> function::Concat::backward(const Variable& gy) {
	size_t n = split_sizes.size();
	const auto& gy_shape = gy.data().get_shape();
	size_t ndim = gy_shape.size();
	int ax = axis;
	if (ax < 0) ax += static_cast<int>(ndim);

	std::vector<Variable> grads;
	grads.reserve(n);

	size_t start = 0;
	for (size_t i = 0; i < n; i++) {
		size_t end = start + split_sizes[i];
		Variable gi(gy.data().slice(ax, start, end).contiguous());
		grads.push_back(gi);
		start = end;
	}

	return grads;
}


Variable function::Reshape::forward(const std::vector<Variable>& xs) {
	const Tensor<>& x_data = xs[0].data();
	x_shape = x_data.get_shape();

	Tensor<> result = x_data.reshape(shape);
	return Variable(result);
}

std::vector<Variable> function::Reshape::backward(const Variable& gy) {
	return { reshape(gy, x_shape) };
}


Variable function::Transpose::forward(const std::vector<Variable>& xs) {
    const Tensor<>& x_data = xs[0].data();
    Tensor<> result = x_data.transpose(axes);
    return Variable(result);
}

std::vector<Variable> function::Transpose::backward(const Variable& gy) {
	if (axes.empty())
		return {transpose(gy)};

	size_t ndim = axes.size();
	std::vector<size_t> inv_axes(ndim);
	for (size_t i = 0; i < ndim; i++)
		inv_axes[axes[i]] = i;
    return { transpose(gy, inv_axes) };
}


Variable function::Upsample::forward(const std::vector<Variable>& xs) {
	const Tensor<>& x = xs[0].data();
	input_shape = x.get_shape();

	// Input: [N, C, H, W]
	size_t N = input_shape[0];
	size_t C = input_shape[1];
	size_t H = input_shape[2];
	size_t W = input_shape[3];
	size_t sf = scale_factor;
	size_t OH = H * sf;
	size_t OW = W * sf;

	std::vector<size_t> out_shape = {N, C, OH, OW};
	size_t total = N * C * OH * OW;
	std::vector<float> out_data(total);

	#pragma omp parallel for collapse(4)
	for (size_t n = 0; n < N; n++) {
		for (size_t c = 0; c < C; c++) {
			for (size_t oh = 0; oh < OH; oh++) {
				for (size_t ow = 0; ow < OW; ow++) {
					size_t ih = oh / sf;
					size_t iw = ow / sf;
					out_data[n * C * OH * OW + c * OH * OW + oh * OW + ow] =
						x({n, c, ih, iw});
				}
			}
		}
	}

	return Variable(Tensor<>(out_shape, out_data));
}

std::vector<Variable> function::Upsample::backward(const Variable& gy) {
	const Tensor<>& gy_data = gy.data();

	size_t N = input_shape[0];
	size_t C = input_shape[1];
	size_t H = input_shape[2];
	size_t W = input_shape[3];
	size_t sf = scale_factor;
	size_t OH = H * sf;
	size_t OW = W * sf;

	Tensor<> gx(input_shape, 0.0f);

	for (size_t n = 0; n < N; n++) {
		for (size_t c = 0; c < C; c++) {
			for (size_t oh = 0; oh < OH; oh++) {
				for (size_t ow = 0; ow < OW; ow++) {
					size_t ih = oh / sf;
					size_t iw = ow / sf;
					gx({n, c, ih, iw}) +=
						gy_data({n, c, oh, ow});
				}
			}
		}
	}

	return { Variable(gx) };
}

