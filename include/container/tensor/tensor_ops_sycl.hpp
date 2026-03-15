#pragma once
#ifdef USE_SYCL

#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include "config/device_sycl.hpp"
#include "container/tensor/tensor.hpp"
#include "container/tensor/tensor_utils.hpp"

namespace tensor {

// Helper: create result tensor on same device
template<typename T>
Tensor<T> make_device_tensor(std::shared_ptr<dcz::SYCLBuffer<T>> buf,
							 const std::vector<size_t>& shape) {
	std::shared_ptr<dcz::DeviceBuffer<T>> base_buf = std::move(buf);
	auto strides = compute_contiguous_strides(shape);
	return Tensor<T>::from_device(dcz::sycl(0), std::move(base_buf), shape, strides);
}

// ========== Element-wise binary ops ==========

template<typename T>
Tensor<T> add_sycl(const Tensor<T>& a, const Tensor<T>& b) {
	auto& q = dcz::SYCLContext::get().queue();
	size_t n = a.size();

	auto result_buf = std::make_shared<dcz::SYCLBuffer<T>>(n, q);
	const T* a_ptr = a.device_buffer()->device_ptr();
	const T* b_ptr = b.device_buffer()->device_ptr();
	T* r_ptr = result_buf->device_ptr();

	q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
		r_ptr[i] = a_ptr[i] + b_ptr[i];
	}).wait();

	return make_device_tensor(result_buf, a.get_shape());
}

template<typename T>
Tensor<T> sub_sycl(const Tensor<T>& a, const Tensor<T>& b) {
	auto& q = dcz::SYCLContext::get().queue();
	size_t n = a.size();

	auto result_buf = std::make_shared<dcz::SYCLBuffer<T>>(n, q);
	const T* a_ptr = a.device_buffer()->device_ptr();
	const T* b_ptr = b.device_buffer()->device_ptr();
	T* r_ptr = result_buf->device_ptr();

	q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
		r_ptr[i] = a_ptr[i] - b_ptr[i];
	}).wait();

	return make_device_tensor(result_buf, a.get_shape());
}

template<typename T>
Tensor<T> mul_sycl(const Tensor<T>& a, const Tensor<T>& b) {
	auto& q = dcz::SYCLContext::get().queue();
	size_t n = a.size();

	auto result_buf = std::make_shared<dcz::SYCLBuffer<T>>(n, q);
	const T* a_ptr = a.device_buffer()->device_ptr();
	const T* b_ptr = b.device_buffer()->device_ptr();
	T* r_ptr = result_buf->device_ptr();

	q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
		r_ptr[i] = a_ptr[i] * b_ptr[i];
	}).wait();

	return make_device_tensor(result_buf, a.get_shape());
}

template<typename T>
Tensor<T> div_sycl(const Tensor<T>& a, const Tensor<T>& b) {
	auto& q = dcz::SYCLContext::get().queue();
	size_t n = a.size();

	auto result_buf = std::make_shared<dcz::SYCLBuffer<T>>(n, q);
	const T* a_ptr = a.device_buffer()->device_ptr();
	const T* b_ptr = b.device_buffer()->device_ptr();
	T* r_ptr = result_buf->device_ptr();

	q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
		r_ptr[i] = a_ptr[i] / b_ptr[i];
	}).wait();

	return make_device_tensor(result_buf, a.get_shape());
}

template<typename T>
Tensor<T> neg_sycl(const Tensor<T>& a) {
	auto& q = dcz::SYCLContext::get().queue();
	size_t n = a.size();

	auto result_buf = std::make_shared<dcz::SYCLBuffer<T>>(n, q);
	const T* a_ptr = a.device_buffer()->device_ptr();
	T* r_ptr = result_buf->device_ptr();

	q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
		r_ptr[i] = -a_ptr[i];
	}).wait();

	return make_device_tensor(result_buf, a.get_shape());
}

// ========== Unary math ops (oneMKL VM) ==========

template<typename T>
Tensor<T> exp_sycl(const Tensor<T>& x) {
	auto& q = dcz::SYCLContext::get().queue();
	size_t n = x.size();
	auto result_buf = std::make_shared<dcz::SYCLBuffer<T>>(n, q);

	oneapi::mkl::vm::exp(q, static_cast<int64_t>(n),
		x.device_buffer()->device_ptr(),
		result_buf->device_ptr()).wait();

	return make_device_tensor(result_buf, x.get_shape());
}

template<typename T>
Tensor<T> log_sycl(const Tensor<T>& x) {
	auto& q = dcz::SYCLContext::get().queue();
	size_t n = x.size();
	auto result_buf = std::make_shared<dcz::SYCLBuffer<T>>(n, q);

	oneapi::mkl::vm::ln(q, static_cast<int64_t>(n),
		x.device_buffer()->device_ptr(),
		result_buf->device_ptr()).wait();

	return make_device_tensor(result_buf, x.get_shape());
}

template<typename T>
Tensor<T> tanh_sycl(const Tensor<T>& x) {
	auto& q = dcz::SYCLContext::get().queue();
	size_t n = x.size();
	auto result_buf = std::make_shared<dcz::SYCLBuffer<T>>(n, q);

	oneapi::mkl::vm::tanh(q, static_cast<int64_t>(n),
		x.device_buffer()->device_ptr(),
		result_buf->device_ptr()).wait();

	return make_device_tensor(result_buf, x.get_shape());
}

template<typename T>
Tensor<T> sin_sycl(const Tensor<T>& x) {
	auto& q = dcz::SYCLContext::get().queue();
	size_t n = x.size();
	auto result_buf = std::make_shared<dcz::SYCLBuffer<T>>(n, q);

	oneapi::mkl::vm::sin(q, static_cast<int64_t>(n),
		x.device_buffer()->device_ptr(),
		result_buf->device_ptr()).wait();

	return make_device_tensor(result_buf, x.get_shape());
}

template<typename T>
Tensor<T> cos_sycl(const Tensor<T>& x) {
	auto& q = dcz::SYCLContext::get().queue();
	size_t n = x.size();
	auto result_buf = std::make_shared<dcz::SYCLBuffer<T>>(n, q);

	oneapi::mkl::vm::cos(q, static_cast<int64_t>(n),
		x.device_buffer()->device_ptr(),
		result_buf->device_ptr()).wait();

	return make_device_tensor(result_buf, x.get_shape());
}

template<typename T>
Tensor<T> pow_sycl(const Tensor<T>& x, const T scalar) {
	auto& q = dcz::SYCLContext::get().queue();
	size_t n = x.size();
	auto result_buf = std::make_shared<dcz::SYCLBuffer<T>>(n, q);

	const T* x_ptr = x.device_buffer()->device_ptr();
	T* r_ptr = result_buf->device_ptr();

	q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
		r_ptr[i] = sycl::pow(x_ptr[i], scalar);
	}).wait();

	return make_device_tensor(result_buf, x.get_shape());
}

// ========== GEMM via oneMKL BLAS ==========

template<typename T>
Tensor<T> dot_sycl(const Tensor<T>& a, const Tensor<T>& b) {
	auto& q = dcz::SYCLContext::get().queue();

	auto a_shape = a.get_shape();
	auto b_shape = b.get_shape();

	if (a_shape.size() < 2 || b_shape.size() < 2)
		throw std::runtime_error("dot_sycl: tensors must be at least 2D");

	// Pad shapes for broadcasting
	size_t ndim = std::max(a_shape.size(), b_shape.size());
	while (a_shape.size() < ndim) a_shape.insert(a_shape.begin(), size_t(1));
	while (b_shape.size() < ndim) b_shape.insert(b_shape.begin(), size_t(1));

	size_t M = a_shape[ndim - 2];
	size_t K = a_shape[ndim - 1];
	size_t N = b_shape[ndim - 1];

	if (K != b_shape[ndim - 2])
		throw std::runtime_error("dot_sycl: inner dimensions mismatch");

	// Compute batch size
	std::vector<size_t> batch_shape;
	for (size_t i = 0; i < ndim - 2; ++i) {
		if (a_shape[i] == b_shape[i]) batch_shape.push_back(a_shape[i]);
		else if (a_shape[i] == 1)     batch_shape.push_back(b_shape[i]);
		else if (b_shape[i] == 1)     batch_shape.push_back(a_shape[i]);
		else throw std::runtime_error("dot_sycl: batch dimension mismatch");
	}

	size_t batch_size = 1;
	for (size_t d : batch_shape) batch_size *= d;

	// Result shape
	std::vector<size_t> result_shape = batch_shape;
	result_shape.push_back(M);
	result_shape.push_back(N);

	size_t total = batch_size * M * N;
	auto result_buf = std::make_shared<dcz::SYCLBuffer<T>>(total, q);

	const T* a_ptr = a.device_buffer()->device_ptr();
	const T* b_ptr = b.device_buffer()->device_ptr();
	T* r_ptr = result_buf->device_ptr();

	// Per-batch GEMM
	for (size_t batch = 0; batch < batch_size; ++batch) {
		const T* a_batch = a_ptr + batch * M * K;
		const T* b_batch = b_ptr + batch * K * N;
		T* r_batch = r_ptr + batch * M * N;

		oneapi::mkl::blas::row_major::gemm(
			q,
			oneapi::mkl::transpose::nontrans,
			oneapi::mkl::transpose::nontrans,
			static_cast<int64_t>(M),
			static_cast<int64_t>(N),
			static_cast<int64_t>(K),
			T(1.0),
			a_batch, static_cast<int64_t>(K),
			b_batch, static_cast<int64_t>(N),
			T(0.0),
			r_batch, static_cast<int64_t>(N)
		).wait();
	}

	return make_device_tensor(result_buf, result_shape);
}

} // namespace tensor

#endif // USE_SYCL
