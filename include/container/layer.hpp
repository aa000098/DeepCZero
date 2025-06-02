#pragma once

#include "container/parameter.hpp"

#include <unordered_map>
#include <string>
#include <memory>

class Parameter;

class Layer {
protected:
	std::unordered_map<std::string, Parameter> params;

public:

	void register_params(const std::string& name, Parameter& param) {
		params[name] = param;
	}

	Parameter get_param(const std::string& name) const {
		auto it = params.find(name);
		if (it != params.end()) it->second;
		else throw std::runtime_error("Parameter not found: " + name);
	}

};
