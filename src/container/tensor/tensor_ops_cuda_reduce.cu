#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cfloat>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call) do { \
	cudaError_t err = (call); \
	if (err != cudaSuccess) \
		throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
} while (0)

static constexpr int BLOCK_SIZE = 256;
static inline int grid_size(size_t n) { return static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE); }

// ========== Axis-based sum reduction ==========
// One thread per output element, iterates over the reduction slice

__global__ void sum_axis_kernel_f(const float* x, float* r,
	const size_t* shape, const bool* is_reduce,
	size_t ndim, size_t total_in, size_t out_total, size_t reduce_size) {
	size_t out_flat = blockIdx.x * blockDim.x + threadIdx.x;
	if (out_flat >= out_total) return;

	// Compute output multi-index for non-reduced dims
	size_t out_rem = out_flat;
	size_t out_idx[8] = {};
	for (size_t d = ndim; d-- > 0;) {
		if (!is_reduce[d]) {
			size_t dim_size = 1;
			for (size_t dd = d + 1; dd < ndim; dd++)
				if (!is_reduce[dd]) dim_size *= shape[dd];
			out_idx[d] = out_rem / dim_size;
			out_rem %= dim_size;
		}
	}

	float sum = 0.0f;
	for (size_t r_idx = 0; r_idx < reduce_size; r_idx++) {
		size_t r_rem = r_idx;
		size_t in_flat = 0;
		size_t stride = total_in;
		for (size_t d = 0; d < ndim; d++) {
			stride /= shape[d];
			size_t idx;
			if (is_reduce[d]) {
				size_t red_stride = 1;
				for (size_t dd = d + 1; dd < ndim; dd++)
					if (is_reduce[dd]) red_stride *= shape[dd];
				idx = r_rem / red_stride;
				r_rem %= red_stride;
			} else {
				idx = out_idx[d];
			}
			in_flat += idx * stride;
		}
		sum += x[in_flat];
	}
	r[out_flat] = sum;
}

// ========== Axis-based max reduction ==========

__global__ void max_axis_kernel_f(const float* x, float* r,
	const size_t* shape, const bool* is_reduce,
	size_t ndim, size_t total_in, size_t out_total, size_t reduce_size) {
	size_t out_flat = blockIdx.x * blockDim.x + threadIdx.x;
	if (out_flat >= out_total) return;

	size_t out_rem = out_flat;
	size_t out_idx[8] = {};
	for (size_t d = ndim; d-- > 0;) {
		if (!is_reduce[d]) {
			size_t dim_size = 1;
			for (size_t dd = d + 1; dd < ndim; dd++)
				if (!is_reduce[dd]) dim_size *= shape[dd];
			out_idx[d] = out_rem / dim_size;
			out_rem %= dim_size;
		}
	}

	float max_val = -FLT_MAX;
	for (size_t r_idx = 0; r_idx < reduce_size; r_idx++) {
		size_t r_rem = r_idx;
		size_t in_flat = 0;
		size_t stride = total_in;
		for (size_t d = 0; d < ndim; d++) {
			stride /= shape[d];
			size_t idx;
			if (is_reduce[d]) {
				size_t red_stride = 1;
				for (size_t dd = d + 1; dd < ndim; dd++)
					if (is_reduce[dd]) red_stride *= shape[dd];
				idx = r_rem / red_stride;
				r_rem %= red_stride;
			} else {
				idx = out_idx[d];
			}
			in_flat += idx * stride;
		}
		float val = x[in_flat];
		if (val > max_val) max_val = val;
	}
	r[out_flat] = max_val;
}

extern "C" {

void cuda_sum_axis_f(const float* x, float* r,
	const size_t* d_shape, const bool* d_is_reduce,
	size_t ndim, size_t total_in, size_t out_total, size_t reduce_size) {
	sum_axis_kernel_f<<<grid_size(out_total), BLOCK_SIZE>>>(
		x, r, d_shape, d_is_reduce, ndim, total_in, out_total, reduce_size);
	CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_max_axis_f(const float* x, float* r,
	const size_t* d_shape, const bool* d_is_reduce,
	size_t ndim, size_t total_in, size_t out_total, size_t reduce_size) {
	max_axis_kernel_f<<<grid_size(out_total), BLOCK_SIZE>>>(
		x, r, d_shape, d_is_reduce, ndim, total_in, out_total, reduce_size);
	CUDA_CHECK(cudaDeviceSynchronize());
}

} // extern "C"

#endif // USE_CUDA
