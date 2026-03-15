#pragma once
#ifdef USE_SYCL

#include <sycl/sycl.hpp>
#include "config/device_sycl.hpp"
#include "container/tensor/tensor.hpp"
#include "container/tensor/tensor_utils.hpp"

namespace tensor {

// ========== Fused activation kernels ==========

template<typename T>
Tensor<T> sigmoid_sycl(const Tensor<T>& x) {
	auto& q = dcz::SYCLContext::get().queue();
	size_t n = x.size();
	auto result_buf = std::make_shared<dcz::SYCLBuffer<T>>(n, q);
	const T* x_ptr = x.device_buffer()->device_ptr();
	T* r_ptr = result_buf->device_ptr();

	q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
		r_ptr[i] = T(1) / (T(1) + sycl::exp(-x_ptr[i]));
	}).wait();

	return make_device_tensor(result_buf, x.get_shape());
}

template<typename T>
Tensor<T> silu_sycl(const Tensor<T>& x) {
	auto& q = dcz::SYCLContext::get().queue();
	size_t n = x.size();
	auto result_buf = std::make_shared<dcz::SYCLBuffer<T>>(n, q);
	const T* x_ptr = x.device_buffer()->device_ptr();
	T* r_ptr = result_buf->device_ptr();

	q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
		T sig = T(1) / (T(1) + sycl::exp(-x_ptr[i]));
		r_ptr[i] = x_ptr[i] * sig;
	}).wait();

	return make_device_tensor(result_buf, x.get_shape());
}

template<typename T>
Tensor<T> relu_sycl(const Tensor<T>& x) {
	auto& q = dcz::SYCLContext::get().queue();
	size_t n = x.size();
	auto result_buf = std::make_shared<dcz::SYCLBuffer<T>>(n, q);
	const T* x_ptr = x.device_buffer()->device_ptr();
	T* r_ptr = result_buf->device_ptr();

	q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
		r_ptr[i] = sycl::max(x_ptr[i], T(0));
	}).wait();

	return make_device_tensor(result_buf, x.get_shape());
}

} // namespace tensor

#endif // USE_SYCL
