#pragma once
#ifdef USE_SYCL

#include <sycl/sycl.hpp>
#include "config/device_sycl.hpp"
#include "container/tensor/tensor.hpp"
#include "container/tensor/tensor_utils.hpp"

namespace tensor {

// ========== Contiguous copy (strided -> contiguous) ==========

template<typename T>
Tensor<T> contiguous_sycl(const Tensor<T>& x) {
	// x is a device tensor with non-contiguous strides
	// We need to copy it to a contiguous layout
	auto& q = dcz::SYCLContext::get().queue();
	auto shape = x.get_shape();
	size_t ndim = shape.size();
	size_t total = 1;
	for (auto s : shape) total *= s;

	auto result_buf = std::make_shared<dcz::SYCLBuffer<T>>(total, q);
	const T* src_ptr = x.device_buffer()->device_ptr();
	T* dst_ptr = result_buf->device_ptr();

	// Copy strides and shape to device-accessible memory
	auto strides = x.get_strides();
	auto out_strides = compute_contiguous_strides(shape);

	// Use USM shared memory for shape/stride arrays (small, kernel needs them)
	size_t* d_shape = sycl::malloc_shared<size_t>(ndim, q);
	size_t* d_strides = sycl::malloc_shared<size_t>(ndim, q);
	for (size_t i = 0; i < ndim; i++) {
		d_shape[i] = shape[i];
		d_strides[i] = strides[i];
	}

	q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> flat) {
		// Compute multi-dim index from flat output index
		size_t rem = flat;
		size_t src_offset = 0;
		for (size_t d = 0; d < ndim; d++) {
			size_t dim_size = 1;
			for (size_t dd = d + 1; dd < ndim; dd++) dim_size *= d_shape[dd];
			size_t idx = rem / dim_size;
			rem %= dim_size;
			src_offset += idx * d_strides[d];
		}
		dst_ptr[flat] = src_ptr[src_offset];
	}).wait();

	sycl::free(d_shape, q);
	sycl::free(d_strides, q);

	return make_device_tensor(result_buf, shape);
}

// ========== Broadcast to target shape ==========

template<typename T>
Tensor<T> broadcast_to_sycl(const Tensor<T>& x, const std::vector<size_t>& target_shape) {
	auto& q = dcz::SYCLContext::get().queue();
	auto src_shape = x.get_shape();
	size_t ndim = target_shape.size();
	size_t total = 1;
	for (auto s : target_shape) total *= s;

	auto result_buf = std::make_shared<dcz::SYCLBuffer<T>>(total, q);
	const T* src_ptr = x.device_buffer()->device_ptr();
	T* dst_ptr = result_buf->device_ptr();

	// Pad source shape to match ndim
	std::vector<size_t> padded_shape(ndim, 1);
	size_t offset = ndim - src_shape.size();
	for (size_t i = 0; i < src_shape.size(); i++)
		padded_shape[offset + i] = src_shape[i];

	// Compute source strides (0 for broadcast dims)
	auto src_strides = x.get_strides();
	std::vector<size_t> padded_strides(ndim, 0);
	for (size_t i = 0; i < src_strides.size(); i++)
		padded_strides[offset + i] = (padded_shape[offset + i] == target_shape[offset + i]) ? src_strides[i] : 0;
	// Leading dims are broadcast (stride=0)

	size_t* d_target = sycl::malloc_shared<size_t>(ndim, q);
	size_t* d_src_strides = sycl::malloc_shared<size_t>(ndim, q);
	for (size_t i = 0; i < ndim; i++) {
		d_target[i] = target_shape[i];
		d_src_strides[i] = padded_strides[i];
	}

	q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> flat) {
		size_t rem = flat;
		size_t src_offset = 0;
		for (size_t d = 0; d < ndim; d++) {
			size_t dim_size = 1;
			for (size_t dd = d + 1; dd < ndim; dd++) dim_size *= d_target[dd];
			size_t idx = rem / dim_size;
			rem %= dim_size;
			src_offset += idx * d_src_strides[d];
		}
		dst_ptr[flat] = src_ptr[src_offset];
	}).wait();

	sycl::free(d_target, q);
	sycl::free(d_src_strides, q);

	return make_device_tensor(result_buf, target_shape);
}

// ========== Inplace ops ==========

template<typename T>
void add_inplace_sycl(Tensor<T>& a, const Tensor<T>& b) {
	auto& q = dcz::SYCLContext::get().queue();
	size_t n = a.size();
	T* a_ptr = a.device_buffer()->device_ptr();
	const T* b_ptr = b.device_buffer()->device_ptr();

	q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
		a_ptr[i] += b_ptr[i];
	}).wait();
}

template<typename T>
void sub_inplace_sycl(Tensor<T>& a, const Tensor<T>& b) {
	auto& q = dcz::SYCLContext::get().queue();
	size_t n = a.size();
	T* a_ptr = a.device_buffer()->device_ptr();
	const T* b_ptr = b.device_buffer()->device_ptr();

	q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
		a_ptr[i] -= b_ptr[i];
	}).wait();
}

template<typename T>
void mul_inplace_sycl(Tensor<T>& a, const Tensor<T>& b) {
	auto& q = dcz::SYCLContext::get().queue();
	size_t n = a.size();
	T* a_ptr = a.device_buffer()->device_ptr();
	const T* b_ptr = b.device_buffer()->device_ptr();

	q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
		a_ptr[i] *= b_ptr[i];
	}).wait();
}

// ========== Scalar ops ==========

template<typename T>
void add_scalar_inplace_sycl(Tensor<T>& a, T scalar) {
	auto& q = dcz::SYCLContext::get().queue();
	size_t n = a.size();
	T* a_ptr = a.device_buffer()->device_ptr();

	q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
		a_ptr[i] += scalar;
	}).wait();
}

template<typename T>
void sub_scalar_inplace_sycl(Tensor<T>& a, T scalar) {
	auto& q = dcz::SYCLContext::get().queue();
	size_t n = a.size();
	T* a_ptr = a.device_buffer()->device_ptr();

	q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
		a_ptr[i] -= scalar;
	}).wait();
}

template<typename T>
void mul_scalar_inplace_sycl(Tensor<T>& a, T scalar) {
	auto& q = dcz::SYCLContext::get().queue();
	size_t n = a.size();
	T* a_ptr = a.device_buffer()->device_ptr();

	q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
		a_ptr[i] *= scalar;
	}).wait();
}

template<typename T>
void div_scalar_inplace_sycl(Tensor<T>& a, T scalar) {
	auto& q = dcz::SYCLContext::get().queue();
	size_t n = a.size();
	T* a_ptr = a.device_buffer()->device_ptr();

	q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
		a_ptr[i] /= scalar;
	}).wait();
}

// ========== Comparison / element-wise ops ==========

template<typename T>
Tensor<T> maximum_sycl(const Tensor<T>& x, T scalar) {
	auto& q = dcz::SYCLContext::get().queue();
	size_t n = x.size();
	auto result_buf = std::make_shared<dcz::SYCLBuffer<T>>(n, q);
	const T* x_ptr = x.device_buffer()->device_ptr();
	T* r_ptr = result_buf->device_ptr();

	q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
		r_ptr[i] = sycl::max(x_ptr[i], scalar);
	}).wait();

	return make_device_tensor(result_buf, x.get_shape());
}

template<typename T>
Tensor<T> minimum_sycl(const Tensor<T>& x, T scalar) {
	auto& q = dcz::SYCLContext::get().queue();
	size_t n = x.size();
	auto result_buf = std::make_shared<dcz::SYCLBuffer<T>>(n, q);
	const T* x_ptr = x.device_buffer()->device_ptr();
	T* r_ptr = result_buf->device_ptr();

	q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
		r_ptr[i] = sycl::min(x_ptr[i], scalar);
	}).wait();

	return make_device_tensor(result_buf, x.get_shape());
}

template<typename T>
Tensor<T> abs_sycl(const Tensor<T>& x) {
	auto& q = dcz::SYCLContext::get().queue();
	size_t n = x.size();
	auto result_buf = std::make_shared<dcz::SYCLBuffer<T>>(n, q);
	const T* x_ptr = x.device_buffer()->device_ptr();
	T* r_ptr = result_buf->device_ptr();

	q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
		r_ptr[i] = sycl::fabs(x_ptr[i]);
	}).wait();

	return make_device_tensor(result_buf, x.get_shape());
}

template<typename T>
Tensor<T> sign_sycl(const Tensor<T>& x) {
	auto& q = dcz::SYCLContext::get().queue();
	size_t n = x.size();
	auto result_buf = std::make_shared<dcz::SYCLBuffer<T>>(n, q);
	const T* x_ptr = x.device_buffer()->device_ptr();
	T* r_ptr = result_buf->device_ptr();

	q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
		T val = x_ptr[i];
		r_ptr[i] = (val > T(0)) ? T(1) : ((val < T(0)) ? T(-1) : T(0));
	}).wait();

	return make_device_tensor(result_buf, x.get_shape());
}

template<typename T>
Tensor<T> clamp_sycl(const Tensor<T>& x, T min_val, T max_val) {
	auto& q = dcz::SYCLContext::get().queue();
	size_t n = x.size();
	auto result_buf = std::make_shared<dcz::SYCLBuffer<T>>(n, q);
	const T* x_ptr = x.device_buffer()->device_ptr();
	T* r_ptr = result_buf->device_ptr();

	q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
		r_ptr[i] = sycl::clamp(x_ptr[i], min_val, max_val);
	}).wait();

	return make_device_tensor(result_buf, x.get_shape());
}

template<typename T>
Tensor<T> greater_sycl(const Tensor<T>& x, T scalar) {
	auto& q = dcz::SYCLContext::get().queue();
	size_t n = x.size();
	auto result_buf = std::make_shared<dcz::SYCLBuffer<T>>(n, q);
	const T* x_ptr = x.device_buffer()->device_ptr();
	T* r_ptr = result_buf->device_ptr();

	q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
		r_ptr[i] = x_ptr[i] > scalar ? T(1) : T(0);
	}).wait();

	return make_device_tensor(result_buf, x.get_shape());
}

// ========== Scalar-tensor reverse ops ==========

template<typename T>
Tensor<T> scalar_sub_sycl(T scalar, const Tensor<T>& x) {
	auto& q = dcz::SYCLContext::get().queue();
	size_t n = x.size();
	auto result_buf = std::make_shared<dcz::SYCLBuffer<T>>(n, q);
	const T* x_ptr = x.device_buffer()->device_ptr();
	T* r_ptr = result_buf->device_ptr();

	q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
		r_ptr[i] = scalar - x_ptr[i];
	}).wait();

	return make_device_tensor(result_buf, x.get_shape());
}

// ========== Broadcasting helper for binary ops ==========

template<typename T>
std::tuple<Tensor<T>, Tensor<T>, std::vector<size_t>>
broadcast_binary_operands_sycl(const Tensor<T>& a, const Tensor<T>& b) {
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
			throw std::runtime_error("broadcast_binary_operands_sycl: shape mismatch");
	}

	Tensor<T> a_bc = (a.get_shape() == shape) ? a : broadcast_to_sycl(a, shape);
	Tensor<T> b_bc = (b.get_shape() == shape) ? b : broadcast_to_sycl(b, shape);
	return {a_bc, b_bc, shape};
}

} // namespace tensor

#endif // USE_SYCL
