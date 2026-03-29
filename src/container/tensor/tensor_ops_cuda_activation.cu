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

__global__ void sigmoid_kernel_f(const float* x, float* r, size_t n) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) r[i] = 1.0f / (1.0f + expf(-x[i]));
}

__global__ void silu_kernel_f(const float* x, float* r, size_t n) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		float sig = 1.0f / (1.0f + expf(-x[i]));
		r[i] = x[i] * sig;
	}
}

__global__ void relu_kernel_f(const float* x, float* r, size_t n) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) r[i] = fmaxf(x[i], 0.0f);
}

extern "C" {

void cuda_sigmoid_f(const float* x, float* r, size_t n) {
	sigmoid_kernel_f<<<grid_size(n), BLOCK_SIZE>>>(x, r, n);
	CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_silu_f(const float* x, float* r, size_t n) {
	silu_kernel_f<<<grid_size(n), BLOCK_SIZE>>>(x, r, n);
	CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_relu_f(const float* x, float* r, size_t n) {
	relu_kernel_f<<<grid_size(n), BLOCK_SIZE>>>(x, r, n);
	CUDA_CHECK(cudaDeviceSynchronize());
}

} // extern "C"

#endif // USE_CUDA
