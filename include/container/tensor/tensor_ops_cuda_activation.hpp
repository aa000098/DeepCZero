#pragma once
#ifdef USE_CUDA

#include "config/device_cuda.hpp"
#include "container/tensor/tensor.hpp"
#include "container/tensor/tensor_utils.hpp"

extern "C" {
void cuda_sigmoid_f(const float* x, float* r, size_t n);
void cuda_silu_f(const float* x, float* r, size_t n);
void cuda_relu_f(const float* x, float* r, size_t n);
}

namespace tensor {

template<typename T>
Tensor<T> sigmoid_cuda(const Tensor<T>& x) {
	size_t n = x.size();
	auto result_buf = std::make_shared<dcz::CUDABuffer<T>>(n);
	if constexpr (std::is_same_v<T, float>)
		cuda_sigmoid_f(x.device_buffer()->device_ptr(), result_buf->device_ptr(), n);
	return make_cuda_tensor(result_buf, x.get_shape());
}

template<typename T>
Tensor<T> silu_cuda(const Tensor<T>& x) {
	size_t n = x.size();
	auto result_buf = std::make_shared<dcz::CUDABuffer<T>>(n);
	if constexpr (std::is_same_v<T, float>)
		cuda_silu_f(x.device_buffer()->device_ptr(), result_buf->device_ptr(), n);
	return make_cuda_tensor(result_buf, x.get_shape());
}

template<typename T>
Tensor<T> relu_cuda(const Tensor<T>& x) {
	size_t n = x.size();
	auto result_buf = std::make_shared<dcz::CUDABuffer<T>>(n);
	if constexpr (std::is_same_v<T, float>)
		cuda_relu_f(x.device_buffer()->device_ptr(), result_buf->device_ptr(), n);
	return make_cuda_tensor(result_buf, x.get_shape());
}

} // namespace tensor

#endif // USE_CUDA
