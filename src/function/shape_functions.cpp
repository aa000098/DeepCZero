#include "function/shape_functions.hpp"
#include "container/tensor/tensor_all.hpp"


Variable function::Concat::forward(const std::vector<Variable>& xs) {
	size_t n = xs.size();
	if (n == 0)
		throw std::runtime_error("Concat: empty input list");

	dcz::Device orig_device = xs[0].device();

	const auto& base_shape = xs[0].data().get_shape();
	size_t ndim = base_shape.size();
	int ax = axis;
	if (ax < 0) ax += static_cast<int>(ndim);

	split_sizes.clear();
	split_sizes.reserve(n);

	std::vector<Tensor<>> tensors;
	tensors.reserve(n);
	for (size_t i = 0; i < n; i++) {
		Tensor<> t = !orig_device.is_cpu() ? xs[i].data().cpu() : xs[i].data();
		tensors.push_back(t);
		split_sizes.push_back(t.get_shape()[ax]);
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

	Tensor<> result(out_shape, out_data);
	if (!orig_device.is_cpu()) result = result.to(orig_device);
	return Variable(result);
}

std::vector<Variable> function::Concat::backward(const Variable& gy) {
	size_t n = split_sizes.size();
	dcz::Device orig_device = gy.device();
	Tensor<> gy_cpu = !orig_device.is_cpu() ? gy.data().cpu() : gy.data();

	size_t ndim = gy_cpu.get_shape().size();
	int ax = axis;
	if (ax < 0) ax += static_cast<int>(ndim);

	std::vector<Variable> grads;
	grads.reserve(n);

	size_t start = 0;
	for (size_t i = 0; i < n; i++) {
		size_t end = start + split_sizes[i];
		Tensor<> gi = gy_cpu.slice(ax, start, end).contiguous();
		if (!orig_device.is_cpu()) gi = gi.to(orig_device);
		grads.push_back(Variable(gi));
		start = end;
	}

	return grads;
}


Variable function::Reshape::forward(const std::vector<Variable>& xs) {
	const Tensor<>& x_data = xs[0].data();
	x_shape = x_data.get_shape();

	// Ensure contiguous before reshape (transpose results may be non-contiguous)
	Tensor<> contiguous_data = x_data.is_contiguous() ? x_data : x_data.contiguous();
	Tensor<> result = contiguous_data.reshape(shape);
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
	dcz::Device orig_device = xs[0].device();
	const Tensor<> x = !orig_device.is_cpu() ? xs[0].data().cpu() : xs[0].data();
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

	Tensor<> result(out_shape, out_data);
	if (!orig_device.is_cpu()) result = result.to(orig_device);
	return Variable(result);
}

std::vector<Variable> function::Upsample::backward(const Variable& gy) {
	dcz::Device orig_device = gy.device();
	const Tensor<> gy_data = !orig_device.is_cpu() ? gy.data().cpu() : gy.data();

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

	if (!orig_device.is_cpu()) {
		Tensor<> result = gx.to(orig_device);
		return { Variable(result) };
	}
	return { Variable(gx) };
}


// Gather: select rows by indices from [N, D] → [N_sel, D]
Variable function::Gather::forward(const std::vector<Variable>& xs) {
	dcz::Device orig_device = xs[0].device();
	const Tensor<> x = !orig_device.is_cpu() ? xs[0].data().cpu() : xs[0].data();
	const auto& shape = x.get_shape();
	src_rows = shape[0];

	size_t D = x.size() / src_rows;
	size_t N_sel = indices.size();
	const auto& x_data = x.raw_data();

	std::vector<float> out_data(N_sel * D);
	for (size_t i = 0; i < N_sel; i++) {
		const float* src = x_data.data() + indices[i] * D;
		std::copy(src, src + D, out_data.data() + i * D);
	}

	std::vector<size_t> out_shape = shape;
	out_shape[0] = N_sel;
	Tensor<> result(out_shape, out_data);
	if (!orig_device.is_cpu()) result = result.to(orig_device);
	return Variable(result);
}

std::vector<Variable> function::Gather::backward(const Variable& gy) {
	dcz::Device orig_device = gy.device();
	Tensor<> gy_cpu = !orig_device.is_cpu() ? gy.data().cpu() : gy.data();
	const auto& gy_data = gy_cpu.raw_data();
	size_t D = gy.size() / indices.size();

	Variable x_orig = inputs[0];
	std::vector<size_t> src_shape = x_orig.shape();
	Tensor<> gx(src_shape, 0.0f);

	for (size_t i = 0; i < indices.size(); i++) {
		for (size_t d = 0; d < D; d++) {
			gx({indices[i], d}) += gy_data[i * D + d];
		}
	}

	if (!orig_device.is_cpu()) gx = gx.to(orig_device);
	return { Variable(gx) };
}

