#pragma once
#ifdef USE_CUDA

#include <cuda_runtime.h>
#include "config/device.hpp"

#define CUDA_CHECK(call) do { \
	cudaError_t err = (call); \
	if (err != cudaSuccess) \
		throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
} while (0)

namespace dcz {

// Singleton CUDA device manager
class CUDAContext {
public:
	static CUDAContext& get() {
		static CUDAContext instance;
		return instance;
	}

	int device_id() const { return device_id_; }

	void print_device_info() const {
		cudaDeviceProp prop;
		CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id_));
		std::cout << "CUDA device: " << prop.name << std::endl;
		std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
		std::cout << "  Global memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
		std::cout << "  SM count: " << prop.multiProcessorCount << std::endl;
	}

private:
	CUDAContext() {
		int count = 0;
		CUDA_CHECK(cudaGetDeviceCount(&count));
		if (count == 0)
			throw std::runtime_error("[CUDA] No CUDA devices found");
		device_id_ = 0;
		CUDA_CHECK(cudaSetDevice(device_id_));
	}
	int device_id_ = 0;
};

// CUDA device memory buffer
template<typename T>
class CUDABuffer : public DeviceBuffer<T> {
public:
	CUDABuffer(size_t count)
		: size_(count) {
		if (count > 0) {
			CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
		}
	}

	~CUDABuffer() override {
		if (ptr_) cudaFree(ptr_);
	}

	// No copy
	CUDABuffer(const CUDABuffer&) = delete;
	CUDABuffer& operator=(const CUDABuffer&) = delete;

	// Move
	CUDABuffer(CUDABuffer&& o) noexcept
		: ptr_(o.ptr_), size_(o.size_) {
		o.ptr_ = nullptr;
		o.size_ = 0;
	}

	size_t size() const override { return size_; }
	Device device() const override { return dcz::cuda(0); }

	std::vector<T> to_host() const override {
		std::vector<T> host(size_);
		CUDA_CHECK(cudaMemcpy(host.data(), ptr_, size_ * sizeof(T), cudaMemcpyDeviceToHost));
		return host;
	}

	void from_host(const T* data, size_t count) override {
		CUDA_CHECK(cudaMemcpy(ptr_, data, count * sizeof(T), cudaMemcpyHostToDevice));
	}

	T* device_ptr() override { return ptr_; }
	const T* device_ptr() const override { return ptr_; }

private:
	T* ptr_ = nullptr;
	size_t size_ = 0;
};

} // namespace dcz

// CPU -> CUDA transfer helper (used by Tensor::to())
namespace tensor {

template<typename T>
Tensor<T> to_cuda(const Tensor<T>& src, const dcz::Device& dev) {
	size_t total = src.size();
	auto shape = src.get_shape();

	auto buf = std::make_shared<dcz::CUDABuffer<T>>(total);

	// Ensure contiguous data for transfer
	if (src.is_contiguous()) {
		buf->from_host(src.raw_data().data(), total);
	} else {
		auto contig = src.contiguous();
		buf->from_host(contig.raw_data().data(), total);
	}

	auto strides = compute_contiguous_strides(shape);
	return Tensor<T>::from_device(dev, buf, shape, strides);
}

} // namespace tensor

#endif // USE_CUDA
