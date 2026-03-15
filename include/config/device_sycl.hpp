#pragma once
#ifdef USE_SYCL

#include <sycl/sycl.hpp>
#include "config/device.hpp"

namespace dcz {

// Singleton SYCL queue manager
class SYCLContext {
public:
	static SYCLContext& get() {
		static SYCLContext instance;
		return instance;
	}

	sycl::queue& queue() { return q_; }

	void print_device_info() const {
		auto dev = q_.get_device();
		std::cout << "SYCL device: " << dev.get_info<sycl::info::device::name>() << std::endl;
		std::cout << "  Max compute units: " << dev.get_info<sycl::info::device::max_compute_units>() << std::endl;
		std::cout << "  Global memory: " << dev.get_info<sycl::info::device::global_mem_size>() / (1024*1024) << " MB" << std::endl;
	}

private:
	SYCLContext() {
		try {
			q_ = sycl::queue(sycl::gpu_selector_v);
		} catch (const std::exception& e) {
			std::cerr << "[SYCL] GPU selector failed (" << e.what() << "), trying default..." << std::endl;
			try {
				q_ = sycl::queue(sycl::default_selector_v);
			} catch (const std::exception& e2) {
				std::cerr << "[SYCL] Default selector failed (" << e2.what() << "), trying CPU..." << std::endl;
				q_ = sycl::queue(sycl::cpu_selector_v);
			}
		}
	}
	sycl::queue q_;
};

// USM-based device buffer for SYCL
template<typename T>
class SYCLBuffer : public DeviceBuffer<T> {
public:
	SYCLBuffer(size_t count, sycl::queue& q)
		: size_(count), q_(q) {
		if (count > 0) {
			usm_ptr_ = sycl::malloc_device<T>(count, q);
			if (!usm_ptr_)
				throw std::runtime_error("SYCLBuffer: failed to allocate "
					+ std::to_string(count * sizeof(T)) + " bytes on device");
		}
	}

	~SYCLBuffer() override {
		if (usm_ptr_) sycl::free(usm_ptr_, q_);
	}

	// No copy
	SYCLBuffer(const SYCLBuffer&) = delete;
	SYCLBuffer& operator=(const SYCLBuffer&) = delete;

	// Move
	SYCLBuffer(SYCLBuffer&& o) noexcept
		: usm_ptr_(o.usm_ptr_), size_(o.size_), q_(o.q_) {
		o.usm_ptr_ = nullptr;
		o.size_ = 0;
	}

	size_t size() const override { return size_; }
	Device device() const override { return dcz::sycl(0); }

	std::vector<T> to_host() const override {
		std::vector<T> host(size_);
		q_.memcpy(host.data(), usm_ptr_, size_ * sizeof(T)).wait();
		return host;
	}

	void from_host(const T* data, size_t count) override {
		q_.memcpy(usm_ptr_, data, count * sizeof(T)).wait();
	}

	T* device_ptr() override { return usm_ptr_; }
	const T* device_ptr() const override { return usm_ptr_; }

private:
	T* usm_ptr_ = nullptr;
	size_t size_ = 0;
	sycl::queue& q_;
};

} // namespace dcz

// CPU -> SYCL transfer helper (used by Tensor::to())
namespace tensor {

template<typename T>
Tensor<T> to_sycl(const Tensor<T>& src, const dcz::Device& dev) {
	auto& q = dcz::SYCLContext::get().queue();
	auto shape = src.get_shape();
	size_t total = src.size();

	auto buf = std::make_shared<dcz::SYCLBuffer<T>>(total, q);

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

#endif // USE_SYCL
