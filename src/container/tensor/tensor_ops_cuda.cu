#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call) do { \
	cudaError_t err = (call); \
	if (err != cudaSuccess) \
		throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
} while (0)

#define CUBLAS_CHECK(call) do { \
	cublasStatus_t stat = (call); \
	if (stat != CUBLAS_STATUS_SUCCESS) \
		throw std::runtime_error("cuBLAS error: " + std::to_string(static_cast<int>(stat))); \
} while (0)

static constexpr int BLOCK_SIZE = 256;

static inline int grid_size(size_t n) {
	return static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
}

// ========== Element-wise binary kernels ==========

__global__ void add_kernel_f(const float* a, const float* b, float* r, size_t n) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) r[i] = a[i] + b[i];
}

__global__ void sub_kernel_f(const float* a, const float* b, float* r, size_t n) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) r[i] = a[i] - b[i];
}

__global__ void mul_kernel_f(const float* a, const float* b, float* r, size_t n) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) r[i] = a[i] * b[i];
}

__global__ void div_kernel_f(const float* a, const float* b, float* r, size_t n) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) r[i] = a[i] / b[i];
}

__global__ void neg_kernel_f(const float* a, float* r, size_t n) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) r[i] = -a[i];
}

// ========== Unary math kernels ==========

__global__ void exp_kernel_f(const float* x, float* r, size_t n) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) r[i] = expf(x[i]);
}

__global__ void log_kernel_f(const float* x, float* r, size_t n) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) r[i] = logf(x[i]);
}

__global__ void tanh_kernel_f(const float* x, float* r, size_t n) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) r[i] = tanhf(x[i]);
}

__global__ void sin_kernel_f(const float* x, float* r, size_t n) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) r[i] = sinf(x[i]);
}

__global__ void cos_kernel_f(const float* x, float* r, size_t n) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) r[i] = cosf(x[i]);
}

__global__ void pow_kernel_f(const float* x, float scalar, float* r, size_t n) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) r[i] = powf(x[i], scalar);
}

// ========== Launcher functions ==========

extern "C" {

void cuda_add_f(const float* a, const float* b, float* r, size_t n) {
	add_kernel_f<<<grid_size(n), BLOCK_SIZE>>>(a, b, r, n);
	CUDA_CHECK(cudaGetLastError());
}

void cuda_sub_f(const float* a, const float* b, float* r, size_t n) {
	sub_kernel_f<<<grid_size(n), BLOCK_SIZE>>>(a, b, r, n);
	CUDA_CHECK(cudaGetLastError());
}

void cuda_mul_f(const float* a, const float* b, float* r, size_t n) {
	mul_kernel_f<<<grid_size(n), BLOCK_SIZE>>>(a, b, r, n);
	CUDA_CHECK(cudaGetLastError());
}

void cuda_div_f(const float* a, const float* b, float* r, size_t n) {
	div_kernel_f<<<grid_size(n), BLOCK_SIZE>>>(a, b, r, n);
	CUDA_CHECK(cudaGetLastError());
}

void cuda_neg_f(const float* a, float* r, size_t n) {
	neg_kernel_f<<<grid_size(n), BLOCK_SIZE>>>(a, r, n);
	CUDA_CHECK(cudaGetLastError());
}

void cuda_exp_f(const float* x, float* r, size_t n) {
	exp_kernel_f<<<grid_size(n), BLOCK_SIZE>>>(x, r, n);
	CUDA_CHECK(cudaGetLastError());
}

void cuda_log_f(const float* x, float* r, size_t n) {
	log_kernel_f<<<grid_size(n), BLOCK_SIZE>>>(x, r, n);
	CUDA_CHECK(cudaGetLastError());
}

void cuda_tanh_f(const float* x, float* r, size_t n) {
	tanh_kernel_f<<<grid_size(n), BLOCK_SIZE>>>(x, r, n);
	CUDA_CHECK(cudaGetLastError());
}

void cuda_sin_f(const float* x, float* r, size_t n) {
	sin_kernel_f<<<grid_size(n), BLOCK_SIZE>>>(x, r, n);
	CUDA_CHECK(cudaGetLastError());
}

void cuda_cos_f(const float* x, float* r, size_t n) {
	cos_kernel_f<<<grid_size(n), BLOCK_SIZE>>>(x, r, n);
	CUDA_CHECK(cudaGetLastError());
}

void cuda_pow_f(const float* x, float scalar, float* r, size_t n) {
	pow_kernel_f<<<grid_size(n), BLOCK_SIZE>>>(x, scalar, r, n);
	CUDA_CHECK(cudaGetLastError());
}

// ========== GEMM via cuBLAS ==========

static cublasHandle_t get_cublas_handle() {
	static cublasHandle_t handle = nullptr;
	if (!handle) {
		CUBLAS_CHECK(cublasCreate(&handle));
	}
	return handle;
}

void cuda_gemm_f(const float* a, const float* b, float* c,
				  size_t M, size_t N, size_t K,
				  float alpha, float beta) {
	cublasHandle_t handle = get_cublas_handle();
	// cuBLAS is column-major, but we can use it for row-major by swapping A and B:
	// C = A * B (row-major) == C^T = B^T * A^T (col-major)
	// So: cublas_gemm(B^T, A^T) with N,M,K swapped
	CUBLAS_CHECK(cublasSgemm(handle,
		CUBLAS_OP_N, CUBLAS_OP_N,
		static_cast<int>(N), static_cast<int>(M), static_cast<int>(K),
		&alpha,
		b, static_cast<int>(N),
		a, static_cast<int>(K),
		&beta,
		c, static_cast<int>(N)));
	CUDA_CHECK(cudaGetLastError());
}

void cuda_gemm_strided_f(bool trans_a, bool trans_b,
						 size_t M, size_t N, size_t K,
						 const float* a, size_t lda,
						 const float* b, size_t ldb,
						 float* c, size_t ldc,
						 const size_t* a_offsets, const size_t* b_offsets, const size_t* c_offsets,
						 const unsigned char* accumulate, size_t count) {
	cublasHandle_t handle = get_cublas_handle();
	const float alpha = 1.0f;
	for (size_t i = 0; i < count; ++i) {
		const float beta = accumulate[i] ? 1.0f : 0.0f;
		CUBLAS_CHECK(cublasSgemm(handle,
			trans_b ? CUBLAS_OP_T : CUBLAS_OP_N,
			trans_a ? CUBLAS_OP_T : CUBLAS_OP_N,
			static_cast<int>(N), static_cast<int>(M), static_cast<int>(K),
			&alpha,
			b + b_offsets[i], static_cast<int>(ldb),
			a + a_offsets[i], static_cast<int>(lda),
			&beta,
			c + c_offsets[i], static_cast<int>(ldc)));
	}
	CUDA_CHECK(cudaGetLastError());
}

} // extern "C"

#endif // USE_CUDA
