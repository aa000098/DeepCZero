#pragma once

namespace dcz {

class BackendConfig {
public:
	bool use_mkl = true;

	static BackendConfig& get() {
		static BackendConfig instance;
		return instance;
	}
};

}
