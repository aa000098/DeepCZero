#pragma once
#ifdef USE_CUDA

#include <cuda_runtime.h>
#include "config/device_cuda.hpp"
#include "container/tensor/tensor.hpp"
#include "container/tensor/tensor_utils.hpp"

extern "C" {
void cuda_contiguous_f(const float* src, float* dst,
	const size_t* d_shape, const size_t* d_strides, size_t ndim, size_t total);
void cuda_broadcast_to_f(const float* src, float* dst,
	const size_t* d_target, const size_t* d_src_strides, size_t ndim, size_t total);
void cuda_add_inplace_f(float* a, const float* b, size_t n);
void cuda_sub_inplace_f(float* a, const float* b, size_t n);
void cuda_mul_inplace_f(float* a, const float* b, size_t n);
void cuda_add_scalar_inplace_f(float* a, float scalar, size_t n);
void cuda_sub_scalar_inplace_f(float* a, float scalar, size_t n);
void cuda_mul_scalar_inplace_f(float* a, float scalar, size_t n);
void cuda_div_scalar_inplace_f(float* a, float scalar, size_t n);
void cuda_maximum_f(const float* x, float scalar, float* r, size_t n);
void cuda_minimum_f(const float* x, float scalar, float* r, size_t n);
void cuda_abs_f(const float* x, float* r, size_t n);
void cuda_sign_f(const float* x, float* r, size_t n);
void cuda_clamp_f(const float* x, float min_val, float max_val, float* r, size_t n);
void cuda_greater_f(const float* x, float scalar, float* r, size_t n);
void cuda_scalar_sub_f(float scalar, const float* x, float* r, size_t n);
void cuda_memcpy_to_device(void* dst, const void* src, size_t bytes);
void* cuda_malloc(size_t bytes);
void cuda_free(void* ptr);
void cuda_memset(void* ptr, int value, size_t bytes);
}

namespace tensor {

// ========== Contiguous copy ==========

template<typename T>
Tensor<T> contiguous_cuda(const Tensor<T>& x) {
	auto shape = x.get_shape();
	size_t ndim = shape.size();
	size_t total = 1;
	for (auto s : shape) total *= s;

	auto result_buf = std::make_shared<dcz::CUDABuffer<T>>(total);
	const T* src_ptr = x.device_buffer()->device_ptr();
	T* dst_ptr = result_buf->device_ptr();

	auto strides = x.get_strides();

	// Copy shape/strides to device
	size_t* d_shape = static_cast<size_t*>(cuda_malloc(ndim * sizeof(size_t)));
	size_t* d_strides = static_cast<size_t*>(cuda_malloc(ndim * sizeof(size_t)));
	cuda_memcpy_to_device(d_shape, shape.data(), ndim * sizeof(size_t));
	cuda_memcpy_to_device(d_strides, strides.data(), ndim * sizeof(size_t));

	if constexpr (std::is_same_v<T, float>)
		cuda_contiguous_f(src_ptr, dst_ptr, d_shape, d_strides, ndim, total);

	cuda_free(d_shape);
	cuda_free(d_strides);

	return make_cuda_tensor(result_buf, shape);
}

// ========== Broadcast to target shape ==========

template<typename T>
Tensor<T> broadcast_to_cuda(const Tensor<T>& x, const std::vector<size_t>& target_shape) {
	auto src_shape = x.get_shape();
	size_t ndim = target_shape.size();
	size_t total = 1;
	for (auto s : target_shape) total *= s;

	auto result_buf = std::make_shared<dcz::CUDABuffer<T>>(total);
	const T* src_ptr = x.device_buffer()->device_ptr();
	T* dst_ptr = result_buf->device_ptr();

	// Pad source shape
	std::vector<size_t> padded_shape(ndim, 1);
	size_t offset = ndim - src_shape.size();
	for (size_t i = 0; i < src_shape.size(); i++)
		padded_shape[offset + i] = src_shape[i];

	auto src_strides = x.get_strides();
	std::vector<size_t> padded_strides(ndim, 0);
	for (size_t i = 0; i < src_strides.size(); i++)
		padded_strides[offset + i] = (padded_shape[offset + i] == target_shape[offset + i]) ? src_strides[i] : 0;

	size_t* d_target = static_cast<size_t*>(cuda_malloc(ndim * sizeof(size_t)));
	size_t* d_src_strides = static_cast<size_t*>(cuda_malloc(ndim * sizeof(size_t)));
	cuda_memcpy_to_device(d_target, target_shape.data(), ndim * sizeof(size_t));
	cuda_memcpy_to_device(d_src_strides, padded_strides.data(), ndim * sizeof(size_t));

	if constexpr (std::is_same_v<T, float>)
		cuda_broadcast_to_f(src_ptr, dst_ptr, d_target, d_src_strides, ndim, total);

	cuda_free(d_target);
	cuda_free(d_src_strides);

	return make_cuda_tensor(result_buf, target_shape);
}

// ========== Inplace ops ==========

template<typename T>
void add_inplace_cuda(Tensor<T>& a, const Tensor<T>& b) {
	if constexpr (std::is_same_v<T, float>)
		cuda_add_inplace_f(a.device_buffer()->device_ptr(), b.device_buffer()->device_ptr(), a.size());
}

template<typename T>
void sub_inplace_cuda(Tensor<T>& a, const Tensor<T>& b) {
	if constexpr (std::is_same_v<T, float>)
		cuda_sub_inplace_f(a.device_buffer()->device_ptr(), b.device_buffer()->device_ptr(), a.size());
}

template<typename T>
void mul_inplace_cuda(Tensor<T>& a, const Tensor<T>& b) {
	if constexpr (std::is_same_v<T, float>)
		cuda_mul_inplace_f(a.device_buffer()->device_ptr(), b.device_buffer()->device_ptr(), a.size());
}

// ========== Scalar inplace ops ==========

template<typename T>
void add_scalar_inplace_cuda(Tensor<T>& a, T scalar) {
	if constexpr (std::is_same_v<T, float>)
		cuda_add_scalar_inplace_f(a.device_buffer()->device_ptr(), scalar, a.size());
}

template<typename T>
void sub_scalar_inplace_cuda(Tensor<T>& a, T scalar) {
	if constexpr (std::is_same_v<T, float>)
		cuda_sub_scalar_inplace_f(a.device_buffer()->device_ptr(), scalar, a.size());
}

template<typename T>
void mul_scalar_inplace_cuda(Tensor<T>& a, T scalar) {
	if constexpr (std::is_same_v<T, float>)
		cuda_mul_scalar_inplace_f(a.device_buffer()->device_ptr(), scalar, a.size());
}

template<typename T>
void div_scalar_inplace_cuda(Tensor<T>& a, T scalar) {
	if constexpr (std::is_same_v<T, float>)
		cuda_div_scalar_inplace_f(a.device_buffer()->device_ptr(), scalar, a.size());
}

// ========== Comparison / element-wise ops ==========

template<typename T>
Tensor<T> maximum_cuda(const Tensor<T>& x, T scalar) {
	size_t n = x.size();
	auto result_buf = std::make_shared<dcz::CUDABuffer<T>>(n);
	if constexpr (std::is_same_v<T, float>)
		cuda_maximum_f(x.device_buffer()->device_ptr(), scalar, result_buf->device_ptr(), n);
	return make_cuda_tensor(result_buf, x.get_shape());
}

template<typename T>
Tensor<T> minimum_cuda(const Tensor<T>& x, T scalar) {
	size_t n = x.size();
	auto result_buf = std::make_shared<dcz::CUDABuffer<T>>(n);
	if constexpr (std::is_same_v<T, float>)
		cuda_minimum_f(x.device_buffer()->device_ptr(), scalar, result_buf->device_ptr(), n);
	return make_cuda_tensor(result_buf, x.get_shape());
}

template<typename T>
Tensor<T> abs_cuda(const Tensor<T>& x) {
	size_t n = x.size();
	auto result_buf = std::make_shared<dcz::CUDABuffer<T>>(n);
	if constexpr (std::is_same_v<T, float>)
		cuda_abs_f(x.device_buffer()->device_ptr(), result_buf->device_ptr(), n);
	return make_cuda_tensor(result_buf, x.get_shape());
}

template<typename T>
Tensor<T> sign_cuda(const Tensor<T>& x) {
	size_t n = x.size();
	auto result_buf = std::make_shared<dcz::CUDABuffer<T>>(n);
	if constexpr (std::is_same_v<T, float>)
		cuda_sign_f(x.device_buffer()->device_ptr(), result_buf->device_ptr(), n);
	return make_cuda_tensor(result_buf, x.get_shape());
}

template<typename T>
Tensor<T> clamp_cuda(const Tensor<T>& x, T min_val, T max_val) {
	size_t n = x.size();
	auto result_buf = std::make_shared<dcz::CUDABuffer<T>>(n);
	if constexpr (std::is_same_v<T, float>)
		cuda_clamp_f(x.device_buffer()->device_ptr(), min_val, max_val, result_buf->device_ptr(), n);
	return make_cuda_tensor(result_buf, x.get_shape());
}

template<typename T>
Tensor<T> greater_cuda(const Tensor<T>& x, T scalar) {
	size_t n = x.size();
	auto result_buf = std::make_shared<dcz::CUDABuffer<T>>(n);
	if constexpr (std::is_same_v<T, float>)
		cuda_greater_f(x.device_buffer()->device_ptr(), scalar, result_buf->device_ptr(), n);
	return make_cuda_tensor(result_buf, x.get_shape());
}

template<typename T>
Tensor<T> scalar_sub_cuda(T scalar, const Tensor<T>& x) {
	size_t n = x.size();
	auto result_buf = std::make_shared<dcz::CUDABuffer<T>>(n);
	if constexpr (std::is_same_v<T, float>)
		cuda_scalar_sub_f(scalar, x.device_buffer()->device_ptr(), result_buf->device_ptr(), n);
	return make_cuda_tensor(result_buf, x.get_shape());
}

// ========== Broadcasting helper for binary ops ==========

template<typename T>
std::tuple<Tensor<T>, Tensor<T>, std::vector<size_t>>
broadcast_binary_operands_cuda(const Tensor<T>& a, const Tensor<T>& b) {
	std::vector<size_t> shape_a = a.get_shape();
	std::vector<size_t> shape_b = b.get_shape();
	size_t ndim = std::max(shape_a.size(), shape_b.size());

	while (shape_a.size() < ndim) shape_a.insert(shape_a.begin(), size_t(1));
	while (shape_b.size() < ndim) shape_b.insert(shape_b.begin(), size_t(1));

	std::vector<size_t> shape(ndim);
	for (size_t i = 0; i < ndim; ++i) {
		if (shape_a[i] == shape_b[i])
			shape[i] = shape_a[i];
		else if (shape_a[i] == 1)
			shape[i] = shape_b[i];
		else if (shape_b[i] == 1)
			shape[i] = shape_a[i];
		else
			throw std::runtime_error("broadcast_binary_operands_cuda: shape mismatch");
	}

	Tensor<T> a_bc = (a.get_shape() == shape) ? a : broadcast_to_cuda(a, shape);
	Tensor<T> b_bc = (b.get_shape() == shape) ? b : broadcast_to_cuda(b, shape);
	return {a_bc, b_bc, shape};
}

} // namespace tensor

#endif // USE_CUDA
