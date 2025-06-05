#pragma once

#include "container/layer/layer.hpp"
#include <vector>
#include <string>

using layer::Layer;

class Model : public Layer {

public:

	Variable forward(const std::vector<Variable>& xs) override {
		(void)xs;
		throw std::runtime_error("Model has no forward"); 
	}
	
	void plot(std::vector<Variable> inputs, std::string to_file="model");

};
