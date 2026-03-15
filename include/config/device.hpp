#pragma once

#include <memory>
#include <vector>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <iostream>

namespace dcz {

enum class DeviceType {
	CPU = 0,
	SYCL = 1,
	// Future backends:
	// CUDA = 2,
	// VULKAN = 3,
};

// Lightweight device descriptor (like torch.device)
class Device {
public:
	DeviceType type;
	int index;

	Device() : type(DeviceType::CPU), index(0) {}
	Device(DeviceType type, int index = 0) : type(type), index(index) {}

	bool is_cpu() const { return type == DeviceType::CPU; }

	bool operator==(const Device& other) const {
		return type == other.type && index == other.index;
	}
	bool operator!=(const Device& other) const { return !(*this == other); }

	std::string str() const {
		switch (type) {
			case DeviceType::CPU:  return "cpu";
			case DeviceType::SYCL: return "sycl:" + std::to_string(index);
			default: return "unknown";
		}
	}
};

inline std::ostream& operator<<(std::ostream& os, const Device& d) {
	return os << d.str();
}

// Convenience device constructors
inline Device cpu() { return Device(DeviceType::CPU); }
inline Device sycl(int index = 0) { return Device(DeviceType::SYCL, index); }

// Abstract base for device-side memory
template<typename T>
class DeviceBuffer {
public:
	virtual ~DeviceBuffer() = default;
	virtual size_t size() const = 0;
	virtual Device device() const = 0;
	virtual std::vector<T> to_host() const = 0;
	virtual void from_host(const T* data, size_t count) = 0;
	virtual T* device_ptr() = 0;
	virtual const T* device_ptr() const = 0;
};

} // namespace dcz
