#pragma once
#ifdef USE_SYCL

#include <sycl/sycl.hpp>
#include <limits>
#include "config/device_sycl.hpp"
#include "container/tensor/tensor.hpp"
#include "container/tensor/tensor_utils.hpp"

namespace tensor {

// ========== Axis-based sum reduction ==========

// Full reduction (sum all elements -> scalar)
template<typename T>
Tensor<T> sum_all_sycl(const Tensor<T>& x) {
	size_t n = x.size();

	// Use host-side reduction for simplicity (transfer once, reduce on CPU)
	// For large tensors, a proper two-pass GPU reduction would be better
	auto host_data = x.device_buffer()->to_host();
	T sum = T(0);
	for (size_t i = 0; i < n; i++) sum += host_data[i];

	Tensor<T> result({1}, sum);
	return result.to(x.device());
}

// Axis-based reduction
template<typename T>
Tensor<T> sum_axis_sycl(const Tensor<T>& x, const std::vector<int>& axes, bool keepdims) {
	auto& q = dcz::SYCLContext::get().queue();
	auto shape = x.get_shape();
	size_t ndim = shape.size();

	// Normalize axes
	std::vector<bool> is_reduce(ndim, false);
	for (int ax : axes) {
		size_t a = (ax < 0) ? ndim + ax : ax;
		is_reduce[a] = true;
	}

	// Compute output shape
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

	// Compute reduction size (product of reduced dims)
	size_t reduce_size = 1;
	for (size_t i = 0; i < ndim; i++)
		if (is_reduce[i]) reduce_size *= shape[i];

	auto result_buf = std::make_shared<dcz::SYCLBuffer<T>>(out_total, q);
	const T* x_ptr = x.device_buffer()->device_ptr();
	T* r_ptr = result_buf->device_ptr();

	// Copy shape info to shared memory for kernel access
	size_t* d_shape = sycl::malloc_shared<size_t>(ndim, q);
	bool* d_is_reduce = sycl::malloc_shared<bool>(ndim, q);
	for (size_t i = 0; i < ndim; i++) {
		d_shape[i] = shape[i];
		d_is_reduce[i] = is_reduce[i];
	}

	size_t total = x.size();

	// Strategy: each output element iterates over its reduction slice
	// One work-item per output element
	q.parallel_for(sycl::range<1>(out_total), [=](sycl::id<1> out_flat) {
		// Compute output multi-index (for non-reduced dims)
		size_t out_rem = out_flat;
		size_t out_idx[8] = {};  // max 8 dims
		// Compute indices for non-reduced dims
		for (size_t d = ndim; d-- > 0;) {
			if (!d_is_reduce[d]) {
				// Compute size of remaining non-reduced dims after this one
				size_t dim_size = 1;
				for (size_t dd = d + 1; dd < ndim; dd++) {
					if (!d_is_reduce[dd]) {
						dim_size *= d_shape[dd];
					}
				}
				out_idx[d] = out_rem / dim_size;
				out_rem %= dim_size;
			}
		}

		// Iterate over all reduced dims
		T sum = T(0);
		for (size_t r = 0; r < reduce_size; r++) {
			// Compute reduced dim indices from flat reduction index
			size_t r_rem = r;
			size_t in_flat = 0;
			size_t stride = total;
			for (size_t d = 0; d < ndim; d++) {
				stride /= d_shape[d];
				size_t idx;
				if (d_is_reduce[d]) {
					// Compute from reduction flat index
					size_t red_stride = 1;
					for (size_t dd = d + 1; dd < ndim; dd++)
						if (d_is_reduce[dd]) red_stride *= d_shape[dd];
					idx = r_rem / red_stride;
					r_rem %= red_stride;
				} else {
					idx = out_idx[d];
				}
				in_flat += idx * stride;
			}
			sum += x_ptr[in_flat];
		}
		r_ptr[out_flat] = sum;
	}).wait();

	sycl::free(d_shape, q);
	sycl::free(d_is_reduce, q);

	return make_device_tensor(result_buf, out_shape);
}

// Dispatcher: routes to full or axis-based reduction
template<typename T>
Tensor<T> sum_sycl(const Tensor<T>& x, const std::vector<int>& axes, bool keepdims) {
	if (axes.empty()) {
		return sum_all_sycl(x);
	}
	return sum_axis_sycl(x, axes, keepdims);
}

// ========== Axis-based max reduction ==========

// Full reduction (max all elements -> scalar)
template<typename T>
Tensor<T> max_all_sycl(const Tensor<T>& x) {
	size_t n = x.size();

	auto host_data = x.device_buffer()->to_host();
	T max_val = host_data[0];
	for (size_t i = 1; i < n; i++)
		if (host_data[i] > max_val) max_val = host_data[i];

	Tensor<T> result({1}, max_val);
	return result.to(x.device());
}

// Axis-based max reduction (cloned from sum_axis_sycl, using max instead of sum)
template<typename T>
Tensor<T> max_axis_sycl(const Tensor<T>& x, const std::vector<int>& axes, bool keepdims) {
	auto& q = dcz::SYCLContext::get().queue();
	auto shape = x.get_shape();
	size_t ndim = shape.size();

	// Normalize axes
	std::vector<bool> is_reduce(ndim, false);
	for (int ax : axes) {
		size_t a = (ax < 0) ? ndim + ax : ax;
		is_reduce[a] = true;
	}

	// Compute output shape
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

	auto result_buf = std::make_shared<dcz::SYCLBuffer<T>>(out_total, q);
	const T* x_ptr = x.device_buffer()->device_ptr();
	T* r_ptr = result_buf->device_ptr();

	size_t* d_shape = sycl::malloc_shared<size_t>(ndim, q);
	bool* d_is_reduce = sycl::malloc_shared<bool>(ndim, q);
	for (size_t i = 0; i < ndim; i++) {
		d_shape[i] = shape[i];
		d_is_reduce[i] = is_reduce[i];
	}

	size_t total = x.size();

	q.parallel_for(sycl::range<1>(out_total), [=](sycl::id<1> out_flat) {
		size_t out_rem = out_flat;
		size_t out_idx[8] = {};
		for (size_t d = ndim; d-- > 0;) {
			if (!d_is_reduce[d]) {
				size_t dim_size = 1;
				for (size_t dd = d + 1; dd < ndim; dd++) {
					if (!d_is_reduce[dd]) {
						dim_size *= d_shape[dd];
					}
				}
				out_idx[d] = out_rem / dim_size;
				out_rem %= dim_size;
			}
		}

		T max_val = -std::numeric_limits<T>::infinity();
		for (size_t r = 0; r < reduce_size; r++) {
			size_t r_rem = r;
			size_t in_flat = 0;
			size_t stride = total;
			for (size_t d = 0; d < ndim; d++) {
				stride /= d_shape[d];
				size_t idx;
				if (d_is_reduce[d]) {
					size_t red_stride = 1;
					for (size_t dd = d + 1; dd < ndim; dd++)
						if (d_is_reduce[dd]) red_stride *= d_shape[dd];
					idx = r_rem / red_stride;
					r_rem %= red_stride;
				} else {
					idx = out_idx[d];
				}
				in_flat += idx * stride;
			}
			T val = x_ptr[in_flat];
			if (val > max_val) max_val = val;
		}
		r_ptr[out_flat] = max_val;
	}).wait();

	sycl::free(d_shape, q);
	sycl::free(d_is_reduce, q);

	return make_device_tensor(result_buf, out_shape);
}

// Dispatcher
template<typename T>
Tensor<T> max_sycl(const Tensor<T>& x, const std::vector<int>& axes, bool keepdims) {
	if (axes.empty()) {
		return max_all_sycl(x);
	}
	return max_axis_sycl(x, axes, keepdims);
}

} // namespace tensor

#endif // USE_SYCL
