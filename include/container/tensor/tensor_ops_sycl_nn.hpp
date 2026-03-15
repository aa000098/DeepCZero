#pragma once
#ifdef USE_SYCL

#include <sycl/sycl.hpp>
#include "config/device_sycl.hpp"
#include "container/tensor/tensor.hpp"
#include "container/tensor/tensor_utils.hpp"

namespace tensor {

// ========== im2col on GPU ==========
// Input: [N, C, H, W] -> Output: [N, C*KH*KW, OH*OW]
// With implicit padding (no separate pad tensor needed)

template<typename T>
Tensor<T> im2col_sycl(const Tensor<T>& img,
					   size_t KH, size_t KW,
					   size_t SH, size_t SW,
					   size_t PH, size_t PW) {
	auto& q = dcz::SYCLContext::get().queue();
	auto shape = img.get_shape();
	size_t N = shape[0], C = shape[1], H = shape[2], W = shape[3];
	size_t OH = (H + 2*PH - KH) / SH + 1;
	size_t OW = (W + 2*PW - KW) / SW + 1;

	size_t col_size = N * C * KH * KW * OH * OW;
	auto result_buf = std::make_shared<dcz::SYCLBuffer<T>>(col_size, q);
	const T* img_ptr = img.device_buffer()->device_ptr();
	T* col_ptr = result_buf->device_ptr();

	q.parallel_for(sycl::range<1>(col_size), [=](sycl::id<1> flat) {
		size_t idx = flat;
		size_t ow = idx % OW; idx /= OW;
		size_t oh = idx % OH; idx /= OH;
		size_t kw = idx % KW; idx /= KW;
		size_t kh = idx % KH; idx /= KH;
		size_t c  = idx % C;  idx /= C;
		size_t n  = idx;

		size_t h_in = kh + oh * SH;
		size_t w_in = kw + ow * SW;

		T val = T(0);
		if (h_in >= PH && h_in < H + PH && w_in >= PW && w_in < W + PW) {
			size_t h_img = h_in - PH;
			size_t w_img = w_in - PW;
			val = img_ptr[n*(C*H*W) + c*(H*W) + h_img*W + w_img];
		}

		// Output layout: [N, C*KH*KW, OH*OW]
		size_t col_c = c * KH * KW + kh * KW + kw;
		size_t col_hw = oh * OW + ow;
		col_ptr[n*(C*KH*KW*OH*OW) + col_c*(OH*OW) + col_hw] = val;
	}).wait();

	return make_device_tensor(result_buf, {N, C*KH*KW, OH*OW});
}

// ========== col2im on GPU ==========
// Reverse of im2col with atomic adds

template<typename T>
Tensor<T> col2im_sycl(const Tensor<T>& col,
					   size_t N, size_t C, size_t H, size_t W,
					   size_t KH, size_t KW,
					   size_t SH, size_t SW,
					   size_t PH, size_t PW) {
	auto& q = dcz::SYCLContext::get().queue();
	size_t OH = (H + 2*PH - KH) / SH + 1;
	size_t OW = (W + 2*PW - KW) / SW + 1;

	size_t img_size = N * C * H * W;
	auto result_buf = std::make_shared<dcz::SYCLBuffer<T>>(img_size, q);
	T* img_ptr = result_buf->device_ptr();

	// Zero-initialize
	q.memset(img_ptr, 0, img_size * sizeof(T)).wait();

	const T* col_ptr = col.device_buffer()->device_ptr();
	size_t col_total = N * C * KH * KW * OH * OW;

	q.parallel_for(sycl::range<1>(col_total), [=](sycl::id<1> flat) {
		size_t idx = flat;
		size_t ow = idx % OW; idx /= OW;
		size_t oh = idx % OH; idx /= OH;
		size_t kw = idx % KW; idx /= KW;
		size_t kh = idx % KH; idx /= KH;
		size_t c  = idx % C;  idx /= C;
		size_t n  = idx;

		size_t h_in = kh + oh * SH;
		size_t w_in = kw + ow * SW;

		if (h_in >= PH && h_in < H + PH && w_in >= PW && w_in < W + PW) {
			size_t h_img = h_in - PH;
			size_t w_img = w_in - PW;
			size_t img_idx = n*(C*H*W) + c*(H*W) + h_img*W + w_img;

			size_t col_c = c * KH * KW + kh * KW + kw;
			size_t col_hw = oh * OW + ow;
			T val = col_ptr[n*(C*KH*KW*OH*OW) + col_c*(OH*OW) + col_hw];

			sycl::atomic_ref<T, sycl::memory_order::relaxed,
				sycl::memory_scope::device,
				sycl::access::address_space::global_space> ref(img_ptr[img_idx]);
			ref.fetch_add(val);
		}
	}).wait();

	return make_device_tensor(result_buf, {N, C, H, W});
}

// ========== MaxPool2d forward ==========

template<typename T>
std::pair<Tensor<T>, Tensor<size_t>> maxpool_forward_sycl(
		const Tensor<T>& x,
		size_t KH, size_t KW,
		size_t SH, size_t SW,
		size_t PH, size_t PW) {
	auto& q = dcz::SYCLContext::get().queue();
	auto shape = x.get_shape();
	size_t N = shape[0], C = shape[1], H = shape[2], W = shape[3];
	size_t OH = (H + 2*PH - KH) / SH + 1;
	size_t OW = (W + 2*PW - KW) / SW + 1;

	size_t out_total = N * C * OH * OW;
	auto out_buf = std::make_shared<dcz::SYCLBuffer<T>>(out_total, q);
	// Argmax indices stored on host (used in backward)
	std::vector<size_t> argmax_data(out_total);

	// For simplicity, use device buffer for argmax too
	size_t* d_argmax = sycl::malloc_device<size_t>(out_total, q);
	const T* x_ptr = x.device_buffer()->device_ptr();
	T* out_ptr = out_buf->device_ptr();

	q.parallel_for(sycl::range<1>(out_total), [=](sycl::id<1> flat) {
		size_t idx = flat;
		size_t ow = idx % OW; idx /= OW;
		size_t oh = idx % OH; idx /= OH;
		size_t c  = idx % C;  idx /= C;
		size_t n  = idx;

		T max_val = -1e30f;
		size_t max_idx = 0;

		for (size_t kh = 0; kh < KH; kh++) {
			for (size_t kw = 0; kw < KW; kw++) {
				size_t h_in = oh * SH + kh;
				size_t w_in = ow * SW + kw;
				if (h_in >= PH && h_in < H + PH && w_in >= PW && w_in < W + PW) {
					size_t h_img = h_in - PH;
					size_t w_img = w_in - PW;
					size_t src_idx = n*(C*H*W) + c*(H*W) + h_img*W + w_img;
					T val = x_ptr[src_idx];
					if (val > max_val) {
						max_val = val;
						max_idx = src_idx;
					}
				}
			}
		}
		out_ptr[flat] = max_val;
		d_argmax[flat] = max_idx;
	}).wait();

	// Transfer argmax to host
	q.memcpy(argmax_data.data(), d_argmax, out_total * sizeof(size_t)).wait();
	sycl::free(d_argmax, q);

	auto out_tensor = make_device_tensor(out_buf, {N, C, OH, OW});
	Tensor<size_t> argmax_tensor({N, C, OH, OW}, argmax_data);

	return {out_tensor, argmax_tensor};
}

// ========== Upsample nearest forward ==========

template<typename T>
Tensor<T> upsample_nearest_sycl(const Tensor<T>& x, size_t scale_h, size_t scale_w) {
	auto& q = dcz::SYCLContext::get().queue();
	auto shape = x.get_shape();
	size_t N = shape[0], C = shape[1], H = shape[2], W = shape[3];
	size_t OH = H * scale_h, OW = W * scale_w;

	size_t total = N * C * OH * OW;
	auto result_buf = std::make_shared<dcz::SYCLBuffer<T>>(total, q);
	const T* x_ptr = x.device_buffer()->device_ptr();
	T* r_ptr = result_buf->device_ptr();

	q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> flat) {
		size_t idx = flat;
		size_t ow = idx % OW; idx /= OW;
		size_t oh = idx % OH; idx /= OH;
		size_t c  = idx % C;  idx /= C;
		size_t n  = idx;

		size_t h_in = oh / scale_h;
		size_t w_in = ow / scale_w;
		r_ptr[flat] = x_ptr[n*(C*H*W) + c*(H*W) + h_in*W + w_in];
	}).wait();

	return make_device_tensor(result_buf, {N, C, OH, OW});
}

// ========== Upsample nearest backward ==========

template<typename T>
Tensor<T> upsample_nearest_backward_sycl(const Tensor<T>& gy,
										   size_t N, size_t C, size_t H, size_t W,
										   size_t scale_h, size_t scale_w) {
	auto& q = dcz::SYCLContext::get().queue();
	size_t OH = H * scale_h, OW = W * scale_w;

	size_t gx_size = N * C * H * W;
	auto result_buf = std::make_shared<dcz::SYCLBuffer<T>>(gx_size, q);
	T* gx_ptr = result_buf->device_ptr();
	q.memset(gx_ptr, 0, gx_size * sizeof(T)).wait();

	const T* gy_ptr = gy.device_buffer()->device_ptr();
	size_t gy_total = N * C * OH * OW;

	q.parallel_for(sycl::range<1>(gy_total), [=](sycl::id<1> flat) {
		size_t idx = flat;
		size_t ow = idx % OW; idx /= OW;
		size_t oh = idx % OH; idx /= OH;
		size_t c  = idx % C;  idx /= C;
		size_t n  = idx;

		size_t h_in = oh / scale_h;
		size_t w_in = ow / scale_w;
		size_t gx_idx = n*(C*H*W) + c*(H*W) + h_in*W + w_in;

		sycl::atomic_ref<T, sycl::memory_order::relaxed,
			sycl::memory_scope::device,
			sycl::access::address_space::global_space> ref(gx_ptr[gx_idx]);
		ref.fetch_add(gy_ptr[flat]);
	}).wait();

	return make_device_tensor(result_buf, {N, C, H, W});
}

// ========== Concat along axis ==========

template<typename T>
Tensor<T> concat_sycl(const std::vector<Tensor<T>>& tensors, size_t axis) {
	auto& q = dcz::SYCLContext::get().queue();

	// Compute output shape
	auto base_shape = tensors[0].get_shape();
	size_t ndim = base_shape.size();
	size_t concat_dim = 0;
	for (auto& t : tensors) concat_dim += t.get_shape()[axis];

	std::vector<size_t> out_shape = base_shape;
	out_shape[axis] = concat_dim;

	size_t total = 1;
	for (auto s : out_shape) total *= s;
	auto result_buf = std::make_shared<dcz::SYCLBuffer<T>>(total, q);
	T* r_ptr = result_buf->device_ptr();

	// Copy each tensor's data sequentially along axis
	size_t axis_offset = 0;
	for (auto& t : tensors) {
		auto t_shape = t.get_shape();
		size_t t_total = t.size();
		const T* t_ptr = t.device_buffer()->device_ptr();

		// Compute outer_size (product of dims before axis)
		// and inner_size (product of dims after axis)
		size_t outer = 1, inner = 1;
		for (size_t d = 0; d < axis; d++) outer *= t_shape[d];
		for (size_t d = axis + 1; d < ndim; d++) inner *= t_shape[d];
		size_t t_axis = t_shape[axis];
		size_t out_axis = out_shape[axis];
		size_t ax_off = axis_offset;

		q.parallel_for(sycl::range<1>(t_total), [=](sycl::id<1> flat) {
			size_t idx = flat;
			size_t inner_idx = idx % inner; idx /= inner;
			size_t ax_idx = idx % t_axis; idx /= t_axis;
			size_t outer_idx = idx;

			size_t out_flat = outer_idx * (out_axis * inner)
							+ (ax_off + ax_idx) * inner
							+ inner_idx;
			r_ptr[out_flat] = t_ptr[flat];
		}).wait();

		axis_offset += t_shape[axis];
	}

	return make_device_tensor(result_buf, out_shape);
}

// ========== Gather rows ==========

template<typename T>
Tensor<T> gather_rows_sycl(const Tensor<T>& x, const std::vector<size_t>& indices) {
	auto& q = dcz::SYCLContext::get().queue();
	auto shape = x.get_shape();
	size_t N = shape[0];
	size_t D = x.size() / N;
	size_t N_sel = indices.size();

	auto result_buf = std::make_shared<dcz::SYCLBuffer<T>>(N_sel * D, q);
	const T* x_ptr = x.device_buffer()->device_ptr();
	T* r_ptr = result_buf->device_ptr();

	// Copy indices to device
	size_t* d_indices = sycl::malloc_device<size_t>(N_sel, q);
	q.memcpy(d_indices, indices.data(), N_sel * sizeof(size_t)).wait();

	q.parallel_for(sycl::range<1>(N_sel * D), [=](sycl::id<1> flat) {
		size_t i = flat / D;
		size_t d = flat % D;
		r_ptr[i * D + d] = x_ptr[d_indices[i] * D + d];
	}).wait();

	sycl::free(d_indices, q);

	std::vector<size_t> out_shape = shape;
	out_shape[0] = N_sel;
	return make_device_tensor(result_buf, out_shape);
}

// ========== Scatter add (gather backward) ==========

template<typename T>
Tensor<T> scatter_add_sycl(const Tensor<T>& grad, const std::vector<size_t>& indices,
						    size_t total_rows, size_t D) {
	auto& q = dcz::SYCLContext::get().queue();
	size_t N_sel = indices.size();

	size_t out_size = total_rows * D;
	auto result_buf = std::make_shared<dcz::SYCLBuffer<T>>(out_size, q);
	T* r_ptr = result_buf->device_ptr();
	q.memset(r_ptr, 0, out_size * sizeof(T)).wait();

	const T* g_ptr = grad.device_buffer()->device_ptr();

	size_t* d_indices = sycl::malloc_device<size_t>(N_sel, q);
	q.memcpy(d_indices, indices.data(), N_sel * sizeof(size_t)).wait();

	q.parallel_for(sycl::range<1>(N_sel * D), [=](sycl::id<1> flat) {
		size_t i = flat / D;
		size_t d = flat % D;
		size_t out_idx = d_indices[i] * D + d;

		sycl::atomic_ref<T, sycl::memory_order::relaxed,
			sycl::memory_scope::device,
			sycl::access::address_space::global_space> ref(r_ptr[out_idx]);
		ref.fetch_add(g_ptr[i * D + d]);
	}).wait();

	sycl::free(d_indices, q);

	return make_device_tensor(result_buf, {total_rows, D});
}

// ========== RMSNorm forward ==========
// x: [batch * seq_len, hidden_size] (flattened first two dims)
// weight: [hidden_size]
// out: [batch * seq_len, hidden_size]

template<typename T>
Tensor<T> rmsnorm_forward_sycl(const Tensor<T>& x, const Tensor<T>& weight, T eps) {
	auto& q = dcz::SYCLContext::get().queue();
	auto shape = x.get_shape();
	size_t rows = 1;
	for (size_t i = 0; i < shape.size() - 1; i++) rows *= shape[i];
	size_t hidden_size = shape.back();

	size_t total = rows * hidden_size;
	auto result_buf = std::make_shared<dcz::SYCLBuffer<T>>(total, q);
	const T* x_ptr = x.device_buffer()->device_ptr();
	const T* w_ptr = weight.device_buffer()->device_ptr();
	T* r_ptr = result_buf->device_ptr();

	// One work-item per row — simple but effective for moderate hidden sizes
	q.parallel_for(sycl::range<1>(rows), [=](sycl::id<1> row) {
		size_t offset = row * hidden_size;

		// Compute mean of x^2
		T sum_sq = T(0);
		for (size_t h = 0; h < hidden_size; h++) {
			T val = x_ptr[offset + h];
			sum_sq += val * val;
		}
		T mean_sq = sum_sq / static_cast<T>(hidden_size);
		T rsqrt_val = sycl::rsqrt(mean_sq + eps);

		// output = x * rsqrt * weight
		for (size_t h = 0; h < hidden_size; h++) {
			r_ptr[offset + h] = x_ptr[offset + h] * rsqrt_val * w_ptr[h];
		}
	}).wait();

	return make_device_tensor(result_buf, shape);
}

// ========== RMSNorm backward ==========
// Returns {dx, dw}

template<typename T>
std::pair<Tensor<T>, Tensor<T>> rmsnorm_backward_sycl(
		const Tensor<T>& x, const Tensor<T>& weight,
		const Tensor<T>& gy, T eps) {
	auto& q = dcz::SYCLContext::get().queue();
	auto shape = x.get_shape();
	size_t rows = 1;
	for (size_t i = 0; i < shape.size() - 1; i++) rows *= shape[i];
	size_t hidden_size = shape.back();

	size_t total = rows * hidden_size;
	auto dx_buf = std::make_shared<dcz::SYCLBuffer<T>>(total, q);
	auto dw_buf = std::make_shared<dcz::SYCLBuffer<T>>(hidden_size, q);

	const T* x_ptr = x.device_buffer()->device_ptr();
	const T* w_ptr = weight.device_buffer()->device_ptr();
	const T* gy_ptr = gy.device_buffer()->device_ptr();
	T* dx_ptr = dx_buf->device_ptr();
	T* dw_ptr = dw_buf->device_ptr();

	// Zero dw
	q.memset(dw_ptr, 0, hidden_size * sizeof(T)).wait();

	T inv_hidden = T(1) / static_cast<T>(hidden_size);

	// Compute dx per row + accumulate dw with atomics
	q.parallel_for(sycl::range<1>(rows), [=](sycl::id<1> row) {
		size_t offset = row * hidden_size;

		// sum_sq
		T sum_sq = T(0);
		for (size_t h = 0; h < hidden_size; h++) {
			T xv = x_ptr[offset + h];
			sum_sq += xv * xv;
		}
		T mean_sq = sum_sq * inv_hidden;
		T inv_rms = sycl::rsqrt(mean_sq + eps);
		T inv_rms3 = inv_rms * inv_rms * inv_rms;

		// dot = sum(gy * w * x)
		T dot = T(0);
		for (size_t h = 0; h < hidden_size; h++) {
			dot += gy_ptr[offset + h] * w_ptr[h] * x_ptr[offset + h];
		}

		for (size_t h = 0; h < hidden_size; h++) {
			T gxw = gy_ptr[offset + h] * w_ptr[h];
			T xv = x_ptr[offset + h];
			dx_ptr[offset + h] = gxw * inv_rms - xv * (inv_rms3 * inv_hidden) * dot;

			// Accumulate dw with atomic add
			T dw_val = gy_ptr[offset + h] * xv * inv_rms;
			sycl::atomic_ref<T, sycl::memory_order::relaxed,
				sycl::memory_scope::device,
				sycl::access::address_space::global_space> ref(dw_ptr[h]);
			ref.fetch_add(dw_val);
		}
	}).wait();

	auto dx = make_device_tensor(dx_buf, shape);
	auto dw = make_device_tensor(dw_buf, weight.get_shape());
	return {dx, dw};
}

// ========== SoftmaxCrossEntropy forward ==========
// x: [N, C], t: [N] (labels as float)
// Returns scalar loss tensor

template<typename T>
Tensor<T> softmax_cross_entropy_forward_sycl(const Tensor<T>& x, const Tensor<T>& t) {
	auto& q = dcz::SYCLContext::get().queue();
	auto shape = x.get_shape();
	size_t N = shape[0];
	size_t C = shape[1];

	const T* x_ptr = x.device_buffer()->device_ptr();
	const T* t_ptr = t.device_buffer()->device_ptr();

	// Per-sample loss buffer
	auto loss_buf = std::make_shared<dcz::SYCLBuffer<T>>(N, q);
	T* loss_ptr = loss_buf->device_ptr();

	// One work-item per sample: compute logsumexp and pick label
	q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
		size_t offset = i * C;

		// Max for numerical stability
		T max_val = x_ptr[offset];
		for (size_t c = 1; c < C; c++) {
			T v = x_ptr[offset + c];
			if (v > max_val) max_val = v;
		}

		// Sum of exp(x - max)
		T sum_exp = T(0);
		for (size_t c = 0; c < C; c++) {
			sum_exp += sycl::exp(x_ptr[offset + c] - max_val);
		}

		T log_z = max_val + sycl::log(sum_exp);
		size_t label = static_cast<size_t>(t_ptr[i]);
		loss_ptr[i] = -(x_ptr[offset + label] - log_z);
	}).wait();

	// Reduce losses on host (small N)
	auto host_loss = loss_buf->to_host();
	T total = T(0);
	for (size_t i = 0; i < N; i++) total += host_loss[i];
	total /= static_cast<T>(N);

	Tensor<T> result({1}, total);
	return result.to(x.device());
}

// ========== SoftmaxCrossEntropy backward ==========
// x: [N, C], t: [N] (labels as float), gy: scalar
// Returns dx: [N, C]

template<typename T>
Tensor<T> softmax_cross_entropy_backward_sycl(
		const Tensor<T>& x, const Tensor<T>& t, T gy_val) {
	auto& q = dcz::SYCLContext::get().queue();
	auto shape = x.get_shape();
	size_t N = shape[0];
	size_t C = shape[1];

	size_t total = N * C;
	auto dx_buf = std::make_shared<dcz::SYCLBuffer<T>>(total, q);
	const T* x_ptr = x.device_buffer()->device_ptr();
	const T* t_ptr = t.device_buffer()->device_ptr();
	T* dx_ptr = dx_buf->device_ptr();

	T scale = gy_val / static_cast<T>(N);

	// One work-item per sample: compute softmax, subtract one-hot, scale
	q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
		size_t offset = i * C;

		// Max for stability
		T max_val = x_ptr[offset];
		for (size_t c = 1; c < C; c++) {
			T v = x_ptr[offset + c];
			if (v > max_val) max_val = v;
		}

		// Softmax
		T sum_exp = T(0);
		for (size_t c = 0; c < C; c++) {
			T e = sycl::exp(x_ptr[offset + c] - max_val);
			dx_ptr[offset + c] = e;
			sum_exp += e;
		}

		size_t label = static_cast<size_t>(t_ptr[i]);
		for (size_t c = 0; c < C; c++) {
			T softmax_val = dx_ptr[offset + c] / sum_exp;
			T one_hot = (c == label) ? T(1) : T(0);
			dx_ptr[offset + c] = (softmax_val - one_hot) * scale;
		}
	}).wait();

	return make_device_tensor(dx_buf, shape);
}

} // namespace tensor

#endif // USE_SYCL
