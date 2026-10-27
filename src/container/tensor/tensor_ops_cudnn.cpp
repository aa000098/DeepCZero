#ifdef USE_CUDNN

#include "container/tensor/tensor_ops_cudnn.hpp"
#include "container/tensor/tensor_ops_cuda.hpp"
#include "config/device_cuda.hpp"

#include <cudnn.h>

#include <array>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

#define CUDNN_CHECK(call) do { \
	cudnnStatus_t status = (call); \
	if (status != CUDNN_STATUS_SUCCESS) \
		throw std::runtime_error(std::string("cuDNN error: ") + cudnnGetErrorString(status)); \
} while (0)

namespace tensor {

namespace {

struct ConvAlgo {
	cudnnConvolutionFwdAlgo_t algo;
	size_t workspace_bytes;
};

struct ConvDescriptors {
	cudnnTensorDescriptor_t x = nullptr;
	cudnnTensorDescriptor_t y = nullptr;
	cudnnFilterDescriptor_t w = nullptr;
	cudnnConvolutionDescriptor_t conv = nullptr;

	ConvDescriptors() {
		CUDNN_CHECK(cudnnCreateTensorDescriptor(&x));
		CUDNN_CHECK(cudnnCreateTensorDescriptor(&y));
		CUDNN_CHECK(cudnnCreateFilterDescriptor(&w));
		CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv));
	}

	~ConvDescriptors() {
		if (conv) cudnnDestroyConvolutionDescriptor(conv);
		if (w) cudnnDestroyFilterDescriptor(w);
		if (y) cudnnDestroyTensorDescriptor(y);
		if (x) cudnnDestroyTensorDescriptor(x);
	}

	ConvDescriptors(const ConvDescriptors&) = delete;
	ConvDescriptors& operator=(const ConvDescriptors&) = delete;
};

cudnnHandle_t cudnn_handle() {
	static cudnnHandle_t handle = [] {
		cudnnHandle_t h;
		CUDNN_CHECK(cudnnCreate(&h));
		return h;
	}();
	return handle;
}

using ConvKey = std::array<size_t, 11>;

ConvAlgo find_forward_algo(const ConvDescriptors& d, const ConvKey& key) {
	static std::mutex mu;
	static std::map<ConvKey, ConvAlgo> cache;

	std::lock_guard<std::mutex> lock(mu);
	auto it = cache.find(key);
	if (it != cache.end())
		return it->second;

	cudnnConvolutionFwdAlgoPerf_t perf[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
	int found = 0;
	CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithm(cudnn_handle(), d.x, d.w, d.conv, d.y,
		CUDNN_CONVOLUTION_FWD_ALGO_COUNT, &found, perf));

	for (int i = 0; i < found; ++i) {
		if (perf[i].status == CUDNN_STATUS_SUCCESS) {
			ConvAlgo algo{perf[i].algo, perf[i].memory};
			cache.emplace(key, algo);
			return algo;
		}
	}
	throw std::runtime_error("cuDNN: no usable forward convolution algorithm");
}

} // namespace

Tensor<float> conv2d_forward_cudnn(const Tensor<float>& x, const Tensor<float>& W,
								   std::pair<size_t, size_t> stride,
								   std::pair<size_t, size_t> pad) {
	auto x_shape = x.get_shape();
	auto w_shape = W.get_shape();
	size_t N = x_shape[0], C = x_shape[1], H = x_shape[2], Wd = x_shape[3];
	size_t OC = w_shape[0], KH = w_shape[2], KW = w_shape[3];

	ConvDescriptors d;
	CUDNN_CHECK(cudnnSetTensor4dDescriptor(d.x, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
		static_cast<int>(N), static_cast<int>(C), static_cast<int>(H), static_cast<int>(Wd)));
	CUDNN_CHECK(cudnnSetFilter4dDescriptor(d.w, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
		static_cast<int>(OC), static_cast<int>(C), static_cast<int>(KH), static_cast<int>(KW)));
	CUDNN_CHECK(cudnnSetConvolution2dDescriptor(d.conv,
		static_cast<int>(pad.first), static_cast<int>(pad.second),
		static_cast<int>(stride.first), static_cast<int>(stride.second),
		1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

	int on, oc, oh, ow;
	CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(d.conv, d.x, d.w, &on, &oc, &oh, &ow));
	CUDNN_CHECK(cudnnSetTensor4dDescriptor(d.y, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, on, oc, oh, ow));

	ConvAlgo algo = find_forward_algo(d, {N, C, H, Wd, OC, KH, KW,
		stride.first, stride.second, pad.first, pad.second});

	std::vector<size_t> out_shape = {static_cast<size_t>(on), static_cast<size_t>(oc),
									 static_cast<size_t>(oh), static_cast<size_t>(ow)};
	auto out_buf = std::make_shared<dcz::CUDABuffer<float>>(product(out_shape));

	std::shared_ptr<dcz::CUDABuffer<unsigned char>> workspace;
	if (algo.workspace_bytes > 0)
		workspace = std::make_shared<dcz::CUDABuffer<unsigned char>>(algo.workspace_bytes);

	const float alpha = 1.0f, beta = 0.0f;
	CUDNN_CHECK(cudnnConvolutionForward(cudnn_handle(), &alpha,
		d.x, x.device_buffer()->device_ptr(),
		d.w, W.device_buffer()->device_ptr(),
		d.conv, algo.algo,
		workspace ? workspace->device_ptr() : nullptr, algo.workspace_bytes,
		&beta, d.y, out_buf->device_ptr()));

	return make_cuda_tensor(out_buf, out_shape);
}

} // namespace tensor

#endif // USE_CUDNN
