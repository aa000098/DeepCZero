#pragma once
#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <limits>
#include "config/device_cuda.hpp"
#include "container/tensor/tensor.hpp"
#include "container/tensor/tensor_utils.hpp"

extern "C" {
void cuda_sum_axis_f(const float* x, float* r,
	const size_t* d_shape, const bool* d_is_reduce,
	size_t ndim, size_t total_in, size_t out_total, size_t reduce_size);
void cuda_max_axis_f(const float* x, float* r,
	const size_t* d_shape, const bool* d_is_reduce,
	size_t ndim, size_t total_in, size_t out_total, size_t reduce_size);
void cuda_memcpy_to_device(void* dst, const void* src, size_t bytes);
void* cuda_malloc(size_t bytes);
void cuda_free(void* ptr);
}

namespace tensor {

// ========== Full reduction (host-side) ==========

template<typename T>
Tensor<T> sum_all_cuda(const Tensor<T>& x) {
	auto host_data = x.device_buffer()->to_host();
	T sum = T(0);
	for (size_t i = 0; i < host_data.size(); i++) sum += host_data[i];
	Tensor<T> result({1}, sum);
	return result.to(x.device());
}

// ========== Axis-based sum ==========

template<typename T>
Tensor<T> sum_axis_cuda(const Tensor<T>& x, const std::vector<int>& axes, bool keepdims) {
	auto shape = x.get_shape();
	size_t ndim = shape.size();

	std::vector<bool> is_reduce(ndim, false);
	for (int ax : axes) {
		size_t a = (ax < 0) ? ndim + ax : ax;
		is_reduce[a] = true;
	}

	std::vector<size_t> out_shape;
	for (size_t i = 0; i < ndim; i++) {
		if (is_reduce[i]) {
			if (keepdims) out_shape.push_back(1);
		} else {
			out_shape.push_back(shape[i]);
		}
	}
	if (out_shape.empty()) out_shape.push_back(1);

	size_t out_total = 1;
	for (auto s : out_shape) out_total *= s;
	size_t reduce_size = 1;
	for (size_t i = 0; i < ndim; i++)
		if (is_reduce[i]) reduce_size *= shape[i];

	auto result_buf = std::make_shared<dcz::CUDABuffer<T>>(out_total);

	// Copy shape/is_reduce to device
	size_t* d_shape = static_cast<size_t*>(cuda_malloc(ndim * sizeof(size_t)));
	bool* d_is_reduce = static_cast<bool*>(cuda_malloc(ndim * sizeof(bool)));
	cuda_memcpy_to_device(d_shape, shape.data(), ndim * sizeof(size_t));
	cuda_memcpy_to_device(d_is_reduce, is_reduce.data(), ndim * sizeof(bool));

	if constexpr (std::is_same_v<T, float>)
		cuda_sum_axis_f(x.device_buffer()->device_ptr(), result_buf->device_ptr(),
			d_shape, d_is_reduce, ndim, x.size(), out_total, reduce_size);

	cuda_free(d_shape);
	cuda_free(d_is_reduce);

	return make_cuda_tensor(result_buf, out_shape);
}

template<typename T>
Tensor<T> sum_cuda(const Tensor<T>& x, const std::vector<int>& axes, bool keepdims) {
	if (axes.empty()) return sum_all_cuda(x);
	return sum_axis_cuda(x, axes, keepdims);
}

// ========== Full max reduction ==========

template<typename T>
Tensor<T> max_all_cuda(const Tensor<T>& x) {
	auto host_data = x.device_buffer()->to_host();
	T max_val = host_data[0];
	for (size_t i = 1; i < host_data.size(); i++)
		if (host_data[i] > max_val) max_val = host_data[i];
	Tensor<T> result({1}, max_val);
	return result.to(x.device());
}

// ========== Axis-based max ==========

template<typename T>
Tensor<T> max_axis_cuda(const Tensor<T>& x, const std::vector<int>& axes, bool keepdims) {
	auto shape = x.get_shape();
	size_t ndim = shape.size();

	std::vector<bool> is_reduce(ndim, false);
	for (int ax : axes) {
		size_t a = (ax < 0) ? ndim + ax : ax;
		is_reduce[a] = true;
	}

	std::vector<size_t> out_shape;
	for (size_t i = 0; i < ndim; i++) {
		if (is_reduce[i]) {
			if (keepdims) out_shape.push_back(1);
		} else {
			out_shape.push_back(shape[i]);
		}
	}
	if (out_shape.empty()) out_shape.push_back(1);

	size_t out_total = 1;
	for (auto s : out_shape) out_total *= s;
	size_t reduce_size = 1;
	for (size_t i = 0; i < ndim; i++)
		if (is_reduce[i]) reduce_size *= shape[i];

	auto result_buf = std::make_shared<dcz::CUDABuffer<T>>(out_total);

	size_t* d_shape = static_cast<size_t*>(cuda_malloc(ndim * sizeof(size_t)));
	bool* d_is_reduce = static_cast<bool*>(cuda_malloc(ndim * sizeof(bool)));
	cuda_memcpy_to_device(d_shape, shape.data(), ndim * sizeof(size_t));
	cuda_memcpy_to_device(d_is_reduce, is_reduce.data(), ndim * sizeof(bool));

	if constexpr (std::is_same_v<T, float>)
		cuda_max_axis_f(x.device_buffer()->device_ptr(), result_buf->device_ptr(),
			d_shape, d_is_reduce, ndim, x.size(), out_total, reduce_size);

	cuda_free(d_shape);
	cuda_free(d_is_reduce);

	return make_cuda_tensor(result_buf, out_shape);
}

template<typename T>
Tensor<T> max_cuda(const Tensor<T>& x, const std::vector<int>& axes, bool keepdims) {
	if (axes.empty()) return max_all_cuda(x);
	return max_axis_cuda(x, axes, keepdims);
}

} // namespace tensor

#endif // USE_CUDA
