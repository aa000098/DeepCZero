#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call) do { \
	cudaError_t err = (call); \
	if (err != cudaSuccess) \
		throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
} while (0)

static constexpr int BLOCK_SIZE = 256;
static inline int grid_size(size_t n) { return static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE); }

// ========== Contiguous copy ==========

__global__ void contiguous_kernel_f(const float* src, float* dst,
	const size_t* shape, const size_t* strides, size_t ndim, size_t total) {
	size_t flat = blockIdx.x * blockDim.x + threadIdx.x;
	if (flat >= total) return;

	size_t rem = flat;
	size_t src_offset = 0;
	for (size_t d = 0; d < ndim; d++) {
		size_t dim_size = 1;
		for (size_t dd = d + 1; dd < ndim; dd++) dim_size *= shape[dd];
		size_t idx = rem / dim_size;
		rem %= dim_size;
		src_offset += idx * strides[d];
	}
	dst[flat] = src[src_offset];
}

// ========== Broadcast to target shape ==========

__global__ void broadcast_to_kernel_f(const float* src, float* dst,
	const size_t* target_shape, const size_t* src_strides, size_t ndim, size_t total) {
	size_t flat = blockIdx.x * blockDim.x + threadIdx.x;
	if (flat >= total) return;

	size_t rem = flat;
	size_t src_offset = 0;
	for (size_t d = 0; d < ndim; d++) {
		size_t dim_size = 1;
		for (size_t dd = d + 1; dd < ndim; dd++) dim_size *= target_shape[dd];
		size_t idx = rem / dim_size;
		rem %= dim_size;
		src_offset += idx * src_strides[d];
	}
	dst[flat] = src[src_offset];
}

// ========== Inplace ops ==========

__global__ void add_inplace_kernel_f(float* a, const float* b, size_t n) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) a[i] += b[i];
}

__global__ void sub_inplace_kernel_f(float* a, const float* b, size_t n) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) a[i] -= b[i];
}

__global__ void mul_inplace_kernel_f(float* a, const float* b, size_t n) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) a[i] *= b[i];
}

// ========== Scalar inplace ops ==========

__global__ void add_scalar_inplace_kernel_f(float* a, float scalar, size_t n) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) a[i] += scalar;
}

__global__ void sub_scalar_inplace_kernel_f(float* a, float scalar, size_t n) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) a[i] -= scalar;
}

__global__ void mul_scalar_inplace_kernel_f(float* a, float scalar, size_t n) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) a[i] *= scalar;
}

__global__ void div_scalar_inplace_kernel_f(float* a, float scalar, size_t n) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) a[i] /= scalar;
}

// ========== Comparison / element-wise ops ==========

__global__ void maximum_kernel_f(const float* x, float scalar, float* r, size_t n) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) r[i] = fmaxf(x[i], scalar);
}

__global__ void minimum_kernel_f(const float* x, float scalar, float* r, size_t n) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) r[i] = fminf(x[i], scalar);
}

__global__ void abs_kernel_f(const float* x, float* r, size_t n) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) r[i] = fabsf(x[i]);
}

__global__ void sign_kernel_f(const float* x, float* r, size_t n) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		float val = x[i];
		r[i] = (val > 0.0f) ? 1.0f : ((val < 0.0f) ? -1.0f : 0.0f);
	}
}

__global__ void clamp_kernel_f(const float* x, float min_val, float max_val, float* r, size_t n) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) r[i] = fminf(fmaxf(x[i], min_val), max_val);
}

__global__ void greater_kernel_f(const float* x, float scalar, float* r, size_t n) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) r[i] = x[i] > scalar ? 1.0f : 0.0f;
}

__global__ void scalar_sub_kernel_f(float scalar, const float* x, float* r, size_t n) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) r[i] = scalar - x[i];
}

// ========== Launcher functions ==========

extern "C" {

void cuda_contiguous_f(const float* src, float* dst,
	const size_t* d_shape, const size_t* d_strides, size_t ndim, size_t total) {
	contiguous_kernel_f<<<grid_size(total), BLOCK_SIZE>>>(src, dst, d_shape, d_strides, ndim, total);
	CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_broadcast_to_f(const float* src, float* dst,
	const size_t* d_target, const size_t* d_src_strides, size_t ndim, size_t total) {
	broadcast_to_kernel_f<<<grid_size(total), BLOCK_SIZE>>>(src, dst, d_target, d_src_strides, ndim, total);
	CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_add_inplace_f(float* a, const float* b, size_t n) {
	add_inplace_kernel_f<<<grid_size(n), BLOCK_SIZE>>>(a, b, n);
	CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_sub_inplace_f(float* a, const float* b, size_t n) {
	sub_inplace_kernel_f<<<grid_size(n), BLOCK_SIZE>>>(a, b, n);
	CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_mul_inplace_f(float* a, const float* b, size_t n) {
	mul_inplace_kernel_f<<<grid_size(n), BLOCK_SIZE>>>(a, b, n);
	CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_add_scalar_inplace_f(float* a, float scalar, size_t n) {
	add_scalar_inplace_kernel_f<<<grid_size(n), BLOCK_SIZE>>>(a, scalar, n);
	CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_sub_scalar_inplace_f(float* a, float scalar, size_t n) {
	sub_scalar_inplace_kernel_f<<<grid_size(n), BLOCK_SIZE>>>(a, scalar, n);
	CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_mul_scalar_inplace_f(float* a, float scalar, size_t n) {
	mul_scalar_inplace_kernel_f<<<grid_size(n), BLOCK_SIZE>>>(a, scalar, n);
	CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_div_scalar_inplace_f(float* a, float scalar, size_t n) {
	div_scalar_inplace_kernel_f<<<grid_size(n), BLOCK_SIZE>>>(a, scalar, n);
	CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_maximum_f(const float* x, float scalar, float* r, size_t n) {
	maximum_kernel_f<<<grid_size(n), BLOCK_SIZE>>>(x, scalar, r, n);
	CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_minimum_f(const float* x, float scalar, float* r, size_t n) {
	minimum_kernel_f<<<grid_size(n), BLOCK_SIZE>>>(x, scalar, r, n);
	CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_abs_f(const float* x, float* r, size_t n) {
	abs_kernel_f<<<grid_size(n), BLOCK_SIZE>>>(x, r, n);
	CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_sign_f(const float* x, float* r, size_t n) {
	sign_kernel_f<<<grid_size(n), BLOCK_SIZE>>>(x, r, n);
	CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_clamp_f(const float* x, float min_val, float max_val, float* r, size_t n) {
	clamp_kernel_f<<<grid_size(n), BLOCK_SIZE>>>(x, min_val, max_val, r, n);
	CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_greater_f(const float* x, float scalar, float* r, size_t n) {
	greater_kernel_f<<<grid_size(n), BLOCK_SIZE>>>(x, scalar, r, n);
	CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_scalar_sub_f(float scalar, const float* x, float* r, size_t n) {
	scalar_sub_kernel_f<<<grid_size(n), BLOCK_SIZE>>>(scalar, x, r, n);
	CUDA_CHECK(cudaDeviceSynchronize());
}

// Helper: copy host array to device (for shape/stride arrays)
void cuda_memcpy_to_device(void* dst, const void* src, size_t bytes) {
	CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
}

void* cuda_malloc(size_t bytes) {
	void* ptr;
	CUDA_CHECK(cudaMalloc(&ptr, bytes));
	return ptr;
}

void cuda_free(void* ptr) {
	cudaFree(ptr);
}

void cuda_memset(void* ptr, int value, size_t bytes) {
	CUDA_CHECK(cudaMemset(ptr, value, bytes));
}

} // extern "C"

#endif // USE_CUDA
