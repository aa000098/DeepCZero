#pragma once

#include <string>
#include <stdexcept>
#include <iostream>

namespace dcz {
	class Config {
	public:
		bool enable_backprop = true;
		bool train = true;

		static Config& get() {
			static Config instance;
			return instance;
		}
	};

	class UsingConfig {
	private:
		std::string name;
		bool old_val;

	public:
		UsingConfig(std::string name, bool new_val) : name(name) {
			if (name == "enable_backprop") {
				old_val = Config::get().enable_backprop;
				Config::get().enable_backprop = new_val;
			} else if (name == "train") {
				old_val = Config::get().train;
				Config::get().train = new_val;
			} else {
				throw std::invalid_argument("Unknown config name: " + name);
			}
		}

		~UsingConfig() {
			if (name == "enable_backprop")
				Config::get().enable_backprop = old_val;
			else if (name == "train")
				Config::get().train = old_val;
		}
	};

	inline UsingConfig no_grad() {
		return UsingConfig("enable_backprop", false);
	}

	inline UsingConfig test_mode() {
		return UsingConfig("train", false);
	}

}
