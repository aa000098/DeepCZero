#pragma once
#ifdef USE_CUDA

#include "config/device_cuda.hpp"
#include "container/tensor/tensor.hpp"
#include "container/tensor/tensor_utils.hpp"

// Extern C launcher declarations
extern "C" {
void cuda_add_f(const float* a, const float* b, float* r, size_t n);
void cuda_sub_f(const float* a, const float* b, float* r, size_t n);
void cuda_mul_f(const float* a, const float* b, float* r, size_t n);
void cuda_div_f(const float* a, const float* b, float* r, size_t n);
void cuda_neg_f(const float* a, float* r, size_t n);
void cuda_exp_f(const float* x, float* r, size_t n);
void cuda_log_f(const float* x, float* r, size_t n);
void cuda_tanh_f(const float* x, float* r, size_t n);
void cuda_sin_f(const float* x, float* r, size_t n);
void cuda_cos_f(const float* x, float* r, size_t n);
void cuda_pow_f(const float* x, float scalar, float* r, size_t n);
void cuda_gemm_f(const float* a, const float* b, float* c,
				  size_t M, size_t N, size_t K, float alpha, float beta);
}

namespace tensor {

// Helper: create result tensor on CUDA device
template<typename T>
Tensor<T> make_cuda_tensor(std::shared_ptr<dcz::CUDABuffer<T>> buf,
						   const std::vector<size_t>& shape) {
	std::shared_ptr<dcz::DeviceBuffer<T>> base_buf = std::move(buf);
	auto strides = compute_contiguous_strides(shape);
	return Tensor<T>::from_device(dcz::cuda(0), std::move(base_buf), shape, strides);
}

// ========== Element-wise binary ops ==========

template<typename T>
Tensor<T> add_cuda(const Tensor<T>& a, const Tensor<T>& b) {
	size_t n = a.size();
	auto result_buf = std::make_shared<dcz::CUDABuffer<T>>(n);
	if constexpr (std::is_same_v<T, float>)
		cuda_add_f(a.device_buffer()->device_ptr(), b.device_buffer()->device_ptr(),
				   result_buf->device_ptr(), n);
	return make_cuda_tensor(result_buf, a.get_shape());
}

template<typename T>
Tensor<T> sub_cuda(const Tensor<T>& a, const Tensor<T>& b) {
	size_t n = a.size();
	auto result_buf = std::make_shared<dcz::CUDABuffer<T>>(n);
	if constexpr (std::is_same_v<T, float>)
		cuda_sub_f(a.device_buffer()->device_ptr(), b.device_buffer()->device_ptr(),
				   result_buf->device_ptr(), n);
	return make_cuda_tensor(result_buf, a.get_shape());
}

template<typename T>
Tensor<T> mul_cuda(const Tensor<T>& a, const Tensor<T>& b) {
	size_t n = a.size();
	auto result_buf = std::make_shared<dcz::CUDABuffer<T>>(n);
	if constexpr (std::is_same_v<T, float>)
		cuda_mul_f(a.device_buffer()->device_ptr(), b.device_buffer()->device_ptr(),
				   result_buf->device_ptr(), n);
	return make_cuda_tensor(result_buf, a.get_shape());
}

template<typename T>
Tensor<T> div_cuda(const Tensor<T>& a, const Tensor<T>& b) {
	size_t n = a.size();
	auto result_buf = std::make_shared<dcz::CUDABuffer<T>>(n);
	if constexpr (std::is_same_v<T, float>)
		cuda_div_f(a.device_buffer()->device_ptr(), b.device_buffer()->device_ptr(),
				   result_buf->device_ptr(), n);
	return make_cuda_tensor(result_buf, a.get_shape());
}

template<typename T>
Tensor<T> neg_cuda(const Tensor<T>& a) {
	size_t n = a.size();
	auto result_buf = std::make_shared<dcz::CUDABuffer<T>>(n);
	if constexpr (std::is_same_v<T, float>)
		cuda_neg_f(a.device_buffer()->device_ptr(), result_buf->device_ptr(), n);
	return make_cuda_tensor(result_buf, a.get_shape());
}

// ========== Unary math ops ==========

template<typename T>
Tensor<T> exp_cuda(const Tensor<T>& x) {
	size_t n = x.size();
	auto result_buf = std::make_shared<dcz::CUDABuffer<T>>(n);
	if constexpr (std::is_same_v<T, float>)
		cuda_exp_f(x.device_buffer()->device_ptr(), result_buf->device_ptr(), n);
	return make_cuda_tensor(result_buf, x.get_shape());
}

template<typename T>
Tensor<T> log_cuda(const Tensor<T>& x) {
	size_t n = x.size();
	auto result_buf = std::make_shared<dcz::CUDABuffer<T>>(n);
	if constexpr (std::is_same_v<T, float>)
		cuda_log_f(x.device_buffer()->device_ptr(), result_buf->device_ptr(), n);
	return make_cuda_tensor(result_buf, x.get_shape());
}

template<typename T>
Tensor<T> tanh_cuda(const Tensor<T>& x) {
	size_t n = x.size();
	auto result_buf = std::make_shared<dcz::CUDABuffer<T>>(n);
	if constexpr (std::is_same_v<T, float>)
		cuda_tanh_f(x.device_buffer()->device_ptr(), result_buf->device_ptr(), n);
	return make_cuda_tensor(result_buf, x.get_shape());
}

template<typename T>
Tensor<T> sin_cuda(const Tensor<T>& x) {
	size_t n = x.size();
	auto result_buf = std::make_shared<dcz::CUDABuffer<T>>(n);
	if constexpr (std::is_same_v<T, float>)
		cuda_sin_f(x.device_buffer()->device_ptr(), result_buf->device_ptr(), n);
	return make_cuda_tensor(result_buf, x.get_shape());
}

template<typename T>
Tensor<T> cos_cuda(const Tensor<T>& x) {
	size_t n = x.size();
	auto result_buf = std::make_shared<dcz::CUDABuffer<T>>(n);
	if constexpr (std::is_same_v<T, float>)
		cuda_cos_f(x.device_buffer()->device_ptr(), result_buf->device_ptr(), n);
	return make_cuda_tensor(result_buf, x.get_shape());
}

template<typename T>
Tensor<T> pow_cuda(const Tensor<T>& x, const T scalar) {
	size_t n = x.size();
	auto result_buf = std::make_shared<dcz::CUDABuffer<T>>(n);
	if constexpr (std::is_same_v<T, float>)
		cuda_pow_f(x.device_buffer()->device_ptr(), scalar, result_buf->device_ptr(), n);
	return make_cuda_tensor(result_buf, x.get_shape());
}

// ========== GEMM via cuBLAS ==========

template<typename T>
Tensor<T> dot_cuda(const Tensor<T>& a, const Tensor<T>& b) {
	auto a_shape = a.get_shape();
	auto b_shape = b.get_shape();

	if (a_shape.size() < 2 || b_shape.size() < 2)
		throw std::runtime_error("dot_cuda: tensors must be at least 2D");

	size_t ndim = std::max(a_shape.size(), b_shape.size());
	while (a_shape.size() < ndim) a_shape.insert(a_shape.begin(), size_t(1));
	while (b_shape.size() < ndim) b_shape.insert(b_shape.begin(), size_t(1));

	size_t M = a_shape[ndim - 2];
	size_t K = a_shape[ndim - 1];
	size_t N = b_shape[ndim - 1];

	if (K != b_shape[ndim - 2])
		throw std::runtime_error("dot_cuda: inner dimensions mismatch");

	// Compute batch size
	std::vector<size_t> batch_shape;
	for (size_t i = 0; i < ndim - 2; ++i) {
		if (a_shape[i] == b_shape[i]) batch_shape.push_back(a_shape[i]);
		else if (a_shape[i] == 1)     batch_shape.push_back(b_shape[i]);
		else if (b_shape[i] == 1)     batch_shape.push_back(a_shape[i]);
		else throw std::runtime_error("dot_cuda: batch dimension mismatch");
	}

	size_t batch_size = 1;
	for (size_t d : batch_shape) batch_size *= d;

	std::vector<size_t> result_shape = batch_shape;
	result_shape.push_back(M);
	result_shape.push_back(N);

	size_t total = batch_size * M * N;
	auto result_buf = std::make_shared<dcz::CUDABuffer<T>>(total);

	const T* a_ptr = a.device_buffer()->device_ptr();
	const T* b_ptr = b.device_buffer()->device_ptr();
	T* r_ptr = result_buf->device_ptr();

	if constexpr (std::is_same_v<T, float>) {
		for (size_t batch = 0; batch < batch_size; ++batch) {
			const float* a_batch = a_ptr + batch * M * K;
			const float* b_batch = b_ptr + batch * K * N;
			float* r_batch = r_ptr + batch * M * N;
			cuda_gemm_f(a_batch, b_batch, r_batch, M, N, K, 1.0f, 0.0f);
		}
	}

	return make_cuda_tensor(result_buf, result_shape);
}

} // namespace tensor

#endif // USE_CUDA
