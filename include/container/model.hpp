#pragma once

#include "container/layer/layer.hpp"
#include <vector>
#include <string>

using layer::Layer;

class Model : public Layer {

public:
	
	void plot(std::vector<Variable> inputs, std::string to_file="model.png");

};
