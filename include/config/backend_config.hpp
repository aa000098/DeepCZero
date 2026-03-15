#pragma once

#include "config/device.hpp"

namespace dcz {

class BackendConfig {
public:
	bool use_mkl = true;
	Device default_device;  // defaults to CPU

	static BackendConfig& get() {
		static BackendConfig instance;
		return instance;
	}
};

}
