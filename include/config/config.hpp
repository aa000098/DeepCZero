#pragma once

namespace dcz {
	class Config {
	public:
		bool enable_backprop = true;

		static Config& get() {
			static Config instance;
			return instance;
		}
	};

	class UsingConfig {
	private:
		bool previous;
	public:
		UsingConfig(bool new_value) {
			previous = Config::get().enable_backprop;
			Config::get().enable_backprop = new_value;
		}

		~UsingConfig() {
			Config::get().enable_backprop = previous;
		}
	};
}
