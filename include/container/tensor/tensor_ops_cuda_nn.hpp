#pragma once
#ifdef USE_CUDA

#include <cuda_runtime.h>
#include "config/device_cuda.hpp"
#include "container/tensor/tensor.hpp"
#include "container/tensor/tensor_utils.hpp"

extern "C" {
void cuda_im2col_f(const float* img, float* col,
	size_t N, size_t C, size_t H, size_t W,
	size_t KH, size_t KW, size_t SH, size_t SW, size_t PH, size_t PW,
	size_t OH, size_t OW);
void cuda_col2im_f(const float* col, float* img,
	size_t N, size_t C, size_t H, size_t W,
	size_t KH, size_t KW, size_t SH, size_t SW, size_t PH, size_t PW,
	size_t OH, size_t OW);
void cuda_maxpool_forward_f(const float* x, float* out, size_t* argmax,
	size_t N, size_t C, size_t H, size_t W,
	size_t KH, size_t KW, size_t SH, size_t SW, size_t PH, size_t PW,
	size_t OH, size_t OW);
void cuda_upsample_nearest_f(const float* x, float* r,
	size_t N, size_t C, size_t H, size_t W,
	size_t scale_h, size_t scale_w);
void cuda_upsample_nearest_bwd_f(const float* gy, float* gx,
	size_t N, size_t C, size_t H, size_t W,
	size_t scale_h, size_t scale_w);
void cuda_concat_f(const float* src, float* dst,
	size_t outer, size_t t_axis, size_t out_axis, size_t inner,
	size_t ax_off, size_t t_total);
void cuda_gather_rows_f(const float* x, float* r,
	const size_t* d_indices, size_t N_sel, size_t D);
void cuda_scatter_add_f(const float* grad, float* r,
	const size_t* d_indices, size_t N_sel, size_t D);
void cuda_rmsnorm_forward_f(const float* x, const float* w, float* r,
	size_t rows, size_t hidden_size, float eps);
void cuda_rmsnorm_backward_f(const float* x, const float* w, const float* gy,
	float* dx, float* dw, size_t rows, size_t hidden_size, float eps);
void cuda_softmax_ce_forward_f(const float* x, const float* t, float* loss,
	size_t N, size_t C);
void cuda_softmax_ce_backward_f(const float* x, const float* t, float* dx,
	float scale, size_t N, size_t C);
void cuda_memcpy_to_device(void* dst, const void* src, size_t bytes);
void* cuda_malloc(size_t bytes);
void cuda_free(void* ptr);
void cuda_memset(void* ptr, int value, size_t bytes);
}

namespace tensor {

// ========== im2col ==========

template<typename T>
Tensor<T> im2col_cuda(const Tensor<T>& img,
					   size_t KH, size_t KW,
					   size_t SH, size_t SW,
					   size_t PH, size_t PW) {
	auto shape = img.get_shape();
	size_t N = shape[0], C = shape[1], H = shape[2], W = shape[3];
	size_t OH = (H + 2*PH - KH) / SH + 1;
	size_t OW = (W + 2*PW - KW) / SW + 1;

	size_t col_size = N * C * KH * KW * OH * OW;
	auto result_buf = std::make_shared<dcz::CUDABuffer<T>>(col_size);

	if constexpr (std::is_same_v<T, float>)
		cuda_im2col_f(img.device_buffer()->device_ptr(), result_buf->device_ptr(),
			N, C, H, W, KH, KW, SH, SW, PH, PW, OH, OW);

	return make_cuda_tensor(result_buf, {N, C*KH*KW, OH*OW});
}

// ========== col2im ==========

template<typename T>
Tensor<T> col2im_cuda(const Tensor<T>& col,
					   size_t N, size_t C, size_t H, size_t W,
					   size_t KH, size_t KW,
					   size_t SH, size_t SW,
					   size_t PH, size_t PW) {
	size_t OH = (H + 2*PH - KH) / SH + 1;
	size_t OW = (W + 2*PW - KW) / SW + 1;

	size_t img_size = N * C * H * W;
	auto result_buf = std::make_shared<dcz::CUDABuffer<T>>(img_size);
	cuda_memset(result_buf->device_ptr(), 0, img_size * sizeof(T));

	if constexpr (std::is_same_v<T, float>)
		cuda_col2im_f(col.device_buffer()->device_ptr(), result_buf->device_ptr(),
			N, C, H, W, KH, KW, SH, SW, PH, PW, OH, OW);

	return make_cuda_tensor(result_buf, {N, C, H, W});
}

// ========== MaxPool2d forward ==========

template<typename T>
std::pair<Tensor<T>, Tensor<size_t>> maxpool_forward_cuda(
		const Tensor<T>& x,
		size_t KH, size_t KW,
		size_t SH, size_t SW,
		size_t PH, size_t PW) {
	auto shape = x.get_shape();
	size_t N = shape[0], C = shape[1], H = shape[2], W = shape[3];
	size_t OH = (H + 2*PH - KH) / SH + 1;
	size_t OW = (W + 2*PW - KW) / SW + 1;

	size_t out_total = N * C * OH * OW;
	auto out_buf = std::make_shared<dcz::CUDABuffer<T>>(out_total);
	std::vector<size_t> argmax_data(out_total);

	size_t* d_argmax = static_cast<size_t*>(cuda_malloc(out_total * sizeof(size_t)));

	if constexpr (std::is_same_v<T, float>)
		cuda_maxpool_forward_f(x.device_buffer()->device_ptr(),
			out_buf->device_ptr(), d_argmax,
			N, C, H, W, KH, KW, SH, SW, PH, PW, OH, OW);

	// Transfer argmax to host
	cudaMemcpy(argmax_data.data(), d_argmax, out_total * sizeof(size_t), cudaMemcpyDeviceToHost);
	cuda_free(d_argmax);

	auto out_tensor = make_cuda_tensor(out_buf, {N, C, OH, OW});
	Tensor<size_t> argmax_tensor({N, C, OH, OW}, argmax_data);

	return {out_tensor, argmax_tensor};
}

// ========== Upsample nearest forward ==========

template<typename T>
Tensor<T> upsample_nearest_cuda(const Tensor<T>& x, size_t scale_h, size_t scale_w) {
	auto shape = x.get_shape();
	size_t N = shape[0], C = shape[1], H = shape[2], W = shape[3];
	size_t OH = H * scale_h, OW = W * scale_w;

	size_t total = N * C * OH * OW;
	auto result_buf = std::make_shared<dcz::CUDABuffer<T>>(total);

	if constexpr (std::is_same_v<T, float>)
		cuda_upsample_nearest_f(x.device_buffer()->device_ptr(),
			result_buf->device_ptr(), N, C, H, W, scale_h, scale_w);

	return make_cuda_tensor(result_buf, {N, C, OH, OW});
}

// ========== Upsample nearest backward ==========

template<typename T>
Tensor<T> upsample_nearest_backward_cuda(const Tensor<T>& gy,
										   size_t N, size_t C, size_t H, size_t W,
										   size_t scale_h, size_t scale_w) {
	size_t gx_size = N * C * H * W;
	auto result_buf = std::make_shared<dcz::CUDABuffer<T>>(gx_size);
	cuda_memset(result_buf->device_ptr(), 0, gx_size * sizeof(T));

	if constexpr (std::is_same_v<T, float>)
		cuda_upsample_nearest_bwd_f(gy.device_buffer()->device_ptr(),
			result_buf->device_ptr(), N, C, H, W, scale_h, scale_w);

	return make_cuda_tensor(result_buf, {N, C, H, W});
}

// ========== Concat along axis ==========

template<typename T>
Tensor<T> concat_cuda(const std::vector<Tensor<T>>& tensors, size_t axis) {
	auto base_shape = tensors[0].get_shape();
	size_t ndim = base_shape.size();
	size_t concat_dim = 0;
	for (auto& t : tensors) concat_dim += t.get_shape()[axis];

	std::vector<size_t> out_shape = base_shape;
	out_shape[axis] = concat_dim;

	size_t total = 1;
	for (auto s : out_shape) total *= s;
	auto result_buf = std::make_shared<dcz::CUDABuffer<T>>(total);
	T* r_ptr = result_buf->device_ptr();

	size_t axis_offset = 0;
	for (auto& t : tensors) {
		auto t_shape = t.get_shape();
		size_t t_total = t.size();
		size_t outer = 1, inner = 1;
		for (size_t d = 0; d < axis; d++) outer *= t_shape[d];
		for (size_t d = axis + 1; d < ndim; d++) inner *= t_shape[d];

		if constexpr (std::is_same_v<T, float>)
			cuda_concat_f(t.device_buffer()->device_ptr(), r_ptr,
				outer, t_shape[axis], out_shape[axis], inner, axis_offset, t_total);

		axis_offset += t_shape[axis];
	}

	return make_cuda_tensor(result_buf, out_shape);
}

// ========== Gather rows ==========

template<typename T>
Tensor<T> gather_rows_cuda(const Tensor<T>& x, const std::vector<size_t>& indices) {
	auto shape = x.get_shape();
	size_t N = shape[0];
	size_t D = x.size() / N;
	size_t N_sel = indices.size();

	auto result_buf = std::make_shared<dcz::CUDABuffer<T>>(N_sel * D);

	size_t* d_indices = static_cast<size_t*>(cuda_malloc(N_sel * sizeof(size_t)));
	cuda_memcpy_to_device(d_indices, indices.data(), N_sel * sizeof(size_t));

	if constexpr (std::is_same_v<T, float>)
		cuda_gather_rows_f(x.device_buffer()->device_ptr(),
			result_buf->device_ptr(), d_indices, N_sel, D);

	cuda_free(d_indices);

	std::vector<size_t> out_shape = shape;
	out_shape[0] = N_sel;
	return make_cuda_tensor(result_buf, out_shape);
}

// ========== Scatter add ==========

template<typename T>
Tensor<T> scatter_add_cuda(const Tensor<T>& grad, const std::vector<size_t>& indices,
						    size_t total_rows, size_t D) {
	size_t N_sel = indices.size();
	size_t out_size = total_rows * D;
	auto result_buf = std::make_shared<dcz::CUDABuffer<T>>(out_size);
	cuda_memset(result_buf->device_ptr(), 0, out_size * sizeof(T));

	size_t* d_indices = static_cast<size_t*>(cuda_malloc(N_sel * sizeof(size_t)));
	cuda_memcpy_to_device(d_indices, indices.data(), N_sel * sizeof(size_t));

	if constexpr (std::is_same_v<T, float>)
		cuda_scatter_add_f(grad.device_buffer()->device_ptr(),
			result_buf->device_ptr(), d_indices, N_sel, D);

	cuda_free(d_indices);

	return make_cuda_tensor(result_buf, {total_rows, D});
}

// ========== RMSNorm forward ==========

template<typename T>
Tensor<T> rmsnorm_forward_cuda(const Tensor<T>& x, const Tensor<T>& weight, T eps) {
	auto shape = x.get_shape();
	size_t rows = 1;
	for (size_t i = 0; i < shape.size() - 1; i++) rows *= shape[i];
	size_t hidden_size = shape.back();

	size_t total = rows * hidden_size;
	auto result_buf = std::make_shared<dcz::CUDABuffer<T>>(total);

	if constexpr (std::is_same_v<T, float>)
		cuda_rmsnorm_forward_f(x.device_buffer()->device_ptr(),
			weight.device_buffer()->device_ptr(),
			result_buf->device_ptr(), rows, hidden_size, eps);

	return make_cuda_tensor(result_buf, shape);
}

// ========== RMSNorm backward ==========

template<typename T>
std::pair<Tensor<T>, Tensor<T>> rmsnorm_backward_cuda(
		const Tensor<T>& x, const Tensor<T>& weight,
		const Tensor<T>& gy, T eps) {
	auto shape = x.get_shape();
	size_t rows = 1;
	for (size_t i = 0; i < shape.size() - 1; i++) rows *= shape[i];
	size_t hidden_size = shape.back();

	size_t total = rows * hidden_size;
	auto dx_buf = std::make_shared<dcz::CUDABuffer<T>>(total);
	auto dw_buf = std::make_shared<dcz::CUDABuffer<T>>(hidden_size);
	cuda_memset(dw_buf->device_ptr(), 0, hidden_size * sizeof(T));

	if constexpr (std::is_same_v<T, float>)
		cuda_rmsnorm_backward_f(x.device_buffer()->device_ptr(),
			weight.device_buffer()->device_ptr(),
			gy.device_buffer()->device_ptr(),
			dx_buf->device_ptr(), dw_buf->device_ptr(),
			rows, hidden_size, eps);

	auto dx = make_cuda_tensor(dx_buf, shape);
	auto dw = make_cuda_tensor(dw_buf, weight.get_shape());
	return {dx, dw};
}

// ========== Softmax cross entropy forward ==========

template<typename T>
Tensor<T> softmax_cross_entropy_forward_cuda(const Tensor<T>& x, const Tensor<T>& t) {
	auto shape = x.get_shape();
	size_t N = shape[0];
	size_t C = shape[1];

	auto loss_buf = std::make_shared<dcz::CUDABuffer<T>>(N);

	if constexpr (std::is_same_v<T, float>)
		cuda_softmax_ce_forward_f(x.device_buffer()->device_ptr(),
			t.device_buffer()->device_ptr(),
			loss_buf->device_ptr(), N, C);

	auto host_loss = loss_buf->to_host();
	T total = T(0);
	for (size_t i = 0; i < N; i++) total += host_loss[i];
	total /= static_cast<T>(N);

	Tensor<T> result({1}, total);
	return result.to(x.device());
}

// ========== Softmax cross entropy backward ==========

template<typename T>
Tensor<T> softmax_cross_entropy_backward_cuda(
		const Tensor<T>& x, const Tensor<T>& t, T gy_val) {
	auto shape = x.get_shape();
	size_t N = shape[0];
	size_t C = shape[1];

	size_t total = N * C;
	auto dx_buf = std::make_shared<dcz::CUDABuffer<T>>(total);
	T scale = gy_val / static_cast<T>(N);

	if constexpr (std::is_same_v<T, float>)
		cuda_softmax_ce_backward_f(x.device_buffer()->device_ptr(),
			t.device_buffer()->device_ptr(),
			dx_buf->device_ptr(), scale, N, C);

	return make_cuda_tensor(dx_buf, shape);
}

} // namespace tensor

#endif // USE_CUDA
