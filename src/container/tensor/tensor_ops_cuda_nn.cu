#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cmath>
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

// ========== im2col ==========

__global__ void im2col_kernel_f(const float* img, float* col,
	size_t N, size_t C, size_t H, size_t W,
	size_t KH, size_t KW, size_t SH, size_t SW, size_t PH, size_t PW,
	size_t OH, size_t OW, size_t col_size) {
	size_t flat = blockIdx.x * blockDim.x + threadIdx.x;
	if (flat >= col_size) return;

	size_t idx = flat;
	size_t ow = idx % OW; idx /= OW;
	size_t oh = idx % OH; idx /= OH;
	size_t kw = idx % KW; idx /= KW;
	size_t kh = idx % KH; idx /= KH;
	size_t c  = idx % C;  idx /= C;
	size_t n  = idx;

	size_t h_in = kh + oh * SH;
	size_t w_in = kw + ow * SW;

	float val = 0.0f;
	if (h_in >= PH && h_in < H + PH && w_in >= PW && w_in < W + PW) {
		size_t h_img = h_in - PH;
		size_t w_img = w_in - PW;
		val = img[n*(C*H*W) + c*(H*W) + h_img*W + w_img];
	}

	size_t col_c = c * KH * KW + kh * KW + kw;
	size_t col_hw = oh * OW + ow;
	col[n*(C*KH*KW*OH*OW) + col_c*(OH*OW) + col_hw] = val;
}

// ========== col2im ==========

__global__ void col2im_kernel_f(const float* col, float* img,
	size_t N, size_t C, size_t H, size_t W,
	size_t KH, size_t KW, size_t SH, size_t SW, size_t PH, size_t PW,
	size_t OH, size_t OW, size_t col_total) {
	size_t flat = blockIdx.x * blockDim.x + threadIdx.x;
	if (flat >= col_total) return;

	size_t idx = flat;
	size_t ow = idx % OW; idx /= OW;
	size_t oh = idx % OH; idx /= OH;
	size_t kw = idx % KW; idx /= KW;
	size_t kh = idx % KH; idx /= KH;
	size_t c  = idx % C;  idx /= C;
	size_t n  = idx;

	size_t h_in = kh + oh * SH;
	size_t w_in = kw + ow * SW;

	if (h_in >= PH && h_in < H + PH && w_in >= PW && w_in < W + PW) {
		size_t h_img = h_in - PH;
		size_t w_img = w_in - PW;
		size_t img_idx = n*(C*H*W) + c*(H*W) + h_img*W + w_img;
		size_t col_c = c * KH * KW + kh * KW + kw;
		size_t col_hw = oh * OW + ow;
		float val = col[n*(C*KH*KW*OH*OW) + col_c*(OH*OW) + col_hw];
		atomicAdd(&img[img_idx], val);
	}
}

// ========== MaxPool2d forward ==========

__global__ void maxpool_forward_kernel_f(const float* x, float* out, size_t* argmax,
	size_t N, size_t C, size_t H, size_t W,
	size_t KH, size_t KW, size_t SH, size_t SW, size_t PH, size_t PW,
	size_t OH, size_t OW, size_t out_total) {
	size_t flat = blockIdx.x * blockDim.x + threadIdx.x;
	if (flat >= out_total) return;

	size_t idx = flat;
	size_t ow = idx % OW; idx /= OW;
	size_t oh = idx % OH; idx /= OH;
	size_t c  = idx % C;  idx /= C;
	size_t n  = idx;

	float max_val = -1e30f;
	size_t max_idx = 0;

	for (size_t kh = 0; kh < KH; kh++) {
		for (size_t kw = 0; kw < KW; kw++) {
			size_t h_in = oh * SH + kh;
			size_t w_in = ow * SW + kw;
			if (h_in >= PH && h_in < H + PH && w_in >= PW && w_in < W + PW) {
				size_t h_img = h_in - PH;
				size_t w_img = w_in - PW;
				size_t src_idx = n*(C*H*W) + c*(H*W) + h_img*W + w_img;
				float val = x[src_idx];
				if (val > max_val) {
					max_val = val;
					max_idx = src_idx;
				}
			}
		}
	}
	out[flat] = max_val;
	argmax[flat] = max_idx;
}

// ========== Upsample nearest forward ==========

__global__ void upsample_nearest_kernel_f(const float* x, float* r,
	size_t N, size_t C, size_t H, size_t W,
	size_t OH, size_t OW, size_t scale_h, size_t scale_w, size_t total) {
	size_t flat = blockIdx.x * blockDim.x + threadIdx.x;
	if (flat >= total) return;

	size_t idx = flat;
	size_t ow = idx % OW; idx /= OW;
	size_t oh = idx % OH; idx /= OH;
	size_t c  = idx % C;  idx /= C;
	size_t n  = idx;

	size_t h_in = oh / scale_h;
	size_t w_in = ow / scale_w;
	r[flat] = x[n*(C*H*W) + c*(H*W) + h_in*W + w_in];
}

// ========== Upsample nearest backward ==========

__global__ void upsample_nearest_bwd_kernel_f(const float* gy, float* gx,
	size_t N, size_t C, size_t H, size_t W,
	size_t OH, size_t OW, size_t scale_h, size_t scale_w, size_t gy_total) {
	size_t flat = blockIdx.x * blockDim.x + threadIdx.x;
	if (flat >= gy_total) return;

	size_t idx = flat;
	size_t ow = idx % OW; idx /= OW;
	size_t oh = idx % OH; idx /= OH;
	size_t c  = idx % C;  idx /= C;
	size_t n  = idx;

	size_t h_in = oh / scale_h;
	size_t w_in = ow / scale_w;
	size_t gx_idx = n*(C*H*W) + c*(H*W) + h_in*W + w_in;
	atomicAdd(&gx[gx_idx], gy[flat]);
}

// ========== Concat along axis ==========

__global__ void concat_kernel_f(const float* src, float* dst,
	size_t outer, size_t t_axis, size_t out_axis, size_t inner,
	size_t ax_off, size_t t_total) {
	size_t flat = blockIdx.x * blockDim.x + threadIdx.x;
	if (flat >= t_total) return;

	size_t idx = flat;
	size_t inner_idx = idx % inner; idx /= inner;
	size_t ax_idx = idx % t_axis; idx /= t_axis;
	size_t outer_idx = idx;

	size_t out_flat = outer_idx * (out_axis * inner)
					+ (ax_off + ax_idx) * inner + inner_idx;
	dst[out_flat] = src[flat];
}

// ========== Gather rows ==========

__global__ void gather_rows_kernel_f(const float* x, float* r,
	const size_t* indices, size_t N_sel, size_t D) {
	size_t flat = blockIdx.x * blockDim.x + threadIdx.x;
	if (flat >= N_sel * D) return;
	size_t i = flat / D;
	size_t d = flat % D;
	r[i * D + d] = x[indices[i] * D + d];
}

// ========== Scatter add ==========

__global__ void scatter_add_kernel_f(const float* grad, float* r,
	const size_t* indices, size_t N_sel, size_t D) {
	size_t flat = blockIdx.x * blockDim.x + threadIdx.x;
	if (flat >= N_sel * D) return;
	size_t i = flat / D;
	size_t d = flat % D;
	size_t out_idx = indices[i] * D + d;
	atomicAdd(&r[out_idx], grad[i * D + d]);
}

// ========== RMSNorm forward ==========

__global__ void rmsnorm_forward_kernel_f(const float* x, const float* w, float* r,
	size_t rows, size_t hidden_size, float eps) {
	size_t row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= rows) return;

	size_t offset = row * hidden_size;
	float sum_sq = 0.0f;
	for (size_t h = 0; h < hidden_size; h++) {
		float val = x[offset + h];
		sum_sq += val * val;
	}
	float mean_sq = sum_sq / static_cast<float>(hidden_size);
	float rsqrt_val = rsqrtf(mean_sq + eps);

	for (size_t h = 0; h < hidden_size; h++)
		r[offset + h] = x[offset + h] * rsqrt_val * w[h];
}

// ========== RMSNorm backward ==========

__global__ void rmsnorm_backward_kernel_f(const float* x, const float* w, const float* gy,
	float* dx, float* dw,
	size_t rows, size_t hidden_size, float eps) {
	size_t row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= rows) return;

	size_t offset = row * hidden_size;
	float inv_hidden = 1.0f / static_cast<float>(hidden_size);

	float sum_sq = 0.0f;
	for (size_t h = 0; h < hidden_size; h++) {
		float xv = x[offset + h];
		sum_sq += xv * xv;
	}
	float mean_sq = sum_sq * inv_hidden;
	float inv_rms = rsqrtf(mean_sq + eps);
	float inv_rms3 = inv_rms * inv_rms * inv_rms;

	float dot = 0.0f;
	for (size_t h = 0; h < hidden_size; h++)
		dot += gy[offset + h] * w[h] * x[offset + h];

	for (size_t h = 0; h < hidden_size; h++) {
		float gxw = gy[offset + h] * w[h];
		float xv = x[offset + h];
		dx[offset + h] = gxw * inv_rms - xv * (inv_rms3 * inv_hidden) * dot;

		float dw_val = gy[offset + h] * xv * inv_rms;
		atomicAdd(&dw[h], dw_val);
	}
}

// ========== Softmax cross entropy forward ==========

__global__ void softmax_ce_forward_kernel_f(const float* x, const float* t, float* loss,
	size_t N, size_t C) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;

	size_t offset = i * C;
	float max_val = x[offset];
	for (size_t c = 1; c < C; c++) {
		float v = x[offset + c];
		if (v > max_val) max_val = v;
	}

	float sum_exp = 0.0f;
	for (size_t c = 0; c < C; c++)
		sum_exp += expf(x[offset + c] - max_val);

	float log_z = max_val + logf(sum_exp);
	size_t label = static_cast<size_t>(t[i]);
	loss[i] = -(x[offset + label] - log_z);
}

// ========== Softmax cross entropy backward ==========

__global__ void softmax_ce_backward_kernel_f(const float* x, const float* t, float* dx,
	float scale, size_t N, size_t C) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;

	size_t offset = i * C;
	float max_val = x[offset];
	for (size_t c = 1; c < C; c++) {
		float v = x[offset + c];
		if (v > max_val) max_val = v;
	}

	float sum_exp = 0.0f;
	for (size_t c = 0; c < C; c++) {
		float e = expf(x[offset + c] - max_val);
		dx[offset + c] = e;
		sum_exp += e;
	}

	size_t label = static_cast<size_t>(t[i]);
	for (size_t c = 0; c < C; c++) {
		float softmax_val = dx[offset + c] / sum_exp;
		float one_hot = (c == label) ? 1.0f : 0.0f;
		dx[offset + c] = (softmax_val - one_hot) * scale;
	}
}

// ========== Launcher functions ==========

extern "C" {

void cuda_im2col_f(const float* img, float* col,
	size_t N, size_t C, size_t H, size_t W,
	size_t KH, size_t KW, size_t SH, size_t SW, size_t PH, size_t PW,
	size_t OH, size_t OW) {
	size_t col_size = N * C * KH * KW * OH * OW;
	im2col_kernel_f<<<grid_size(col_size), BLOCK_SIZE>>>(img, col, N, C, H, W, KH, KW, SH, SW, PH, PW, OH, OW, col_size);
	CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_col2im_f(const float* col, float* img,
	size_t N, size_t C, size_t H, size_t W,
	size_t KH, size_t KW, size_t SH, size_t SW, size_t PH, size_t PW,
	size_t OH, size_t OW) {
	size_t col_total = N * C * KH * KW * OH * OW;
	col2im_kernel_f<<<grid_size(col_total), BLOCK_SIZE>>>(col, img, N, C, H, W, KH, KW, SH, SW, PH, PW, OH, OW, col_total);
	CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_maxpool_forward_f(const float* x, float* out, size_t* argmax,
	size_t N, size_t C, size_t H, size_t W,
	size_t KH, size_t KW, size_t SH, size_t SW, size_t PH, size_t PW,
	size_t OH, size_t OW) {
	size_t out_total = N * C * OH * OW;
	maxpool_forward_kernel_f<<<grid_size(out_total), BLOCK_SIZE>>>(x, out, argmax, N, C, H, W, KH, KW, SH, SW, PH, PW, OH, OW, out_total);
	CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_upsample_nearest_f(const float* x, float* r,
	size_t N, size_t C, size_t H, size_t W,
	size_t scale_h, size_t scale_w) {
	size_t OH = H * scale_h, OW = W * scale_w;
	size_t total = N * C * OH * OW;
	upsample_nearest_kernel_f<<<grid_size(total), BLOCK_SIZE>>>(x, r, N, C, H, W, OH, OW, scale_h, scale_w, total);
	CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_upsample_nearest_bwd_f(const float* gy, float* gx,
	size_t N, size_t C, size_t H, size_t W,
	size_t scale_h, size_t scale_w) {
	size_t OH = H * scale_h, OW = W * scale_w;
	size_t gy_total = N * C * OH * OW;
	upsample_nearest_bwd_kernel_f<<<grid_size(gy_total), BLOCK_SIZE>>>(gy, gx, N, C, H, W, OH, OW, scale_h, scale_w, gy_total);
	CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_concat_f(const float* src, float* dst,
	size_t outer, size_t t_axis, size_t out_axis, size_t inner,
	size_t ax_off, size_t t_total) {
	concat_kernel_f<<<grid_size(t_total), BLOCK_SIZE>>>(src, dst, outer, t_axis, out_axis, inner, ax_off, t_total);
	CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_gather_rows_f(const float* x, float* r,
	const size_t* d_indices, size_t N_sel, size_t D) {
	size_t total = N_sel * D;
	gather_rows_kernel_f<<<grid_size(total), BLOCK_SIZE>>>(x, r, d_indices, N_sel, D);
	CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_scatter_add_f(const float* grad, float* r,
	const size_t* d_indices, size_t N_sel, size_t D) {
	size_t total = N_sel * D;
	scatter_add_kernel_f<<<grid_size(total), BLOCK_SIZE>>>(grad, r, d_indices, N_sel, D);
	CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_rmsnorm_forward_f(const float* x, const float* w, float* r,
	size_t rows, size_t hidden_size, float eps) {
	rmsnorm_forward_kernel_f<<<grid_size(rows), BLOCK_SIZE>>>(x, w, r, rows, hidden_size, eps);
	CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_rmsnorm_backward_f(const float* x, const float* w, const float* gy,
	float* dx, float* dw, size_t rows, size_t hidden_size, float eps) {
	rmsnorm_backward_kernel_f<<<grid_size(rows), BLOCK_SIZE>>>(x, w, gy, dx, dw, rows, hidden_size, eps);
	CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_softmax_ce_forward_f(const float* x, const float* t, float* loss,
	size_t N, size_t C) {
	softmax_ce_forward_kernel_f<<<grid_size(N), BLOCK_SIZE>>>(x, t, loss, N, C);
	CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_softmax_ce_backward_f(const float* x, const float* t, float* dx,
	float scale, size_t N, size_t C) {
	softmax_ce_backward_kernel_f<<<grid_size(N), BLOCK_SIZE>>>(x, t, dx, scale, N, C);
	CUDA_CHECK(cudaDeviceSynchronize());
}

} // extern "C"

#endif // USE_CUDA
