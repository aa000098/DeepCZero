#pragma once

#include "container/layer/layer.hpp"
#include "function/function.hpp"
#include "function/activation_functions.hpp"

#include <vector>
#include <string>
#include <memory>

using function::Sigmoid;
using layer::Layer;
using layer::Linear;

class Model : public Layer {

public:

	Variable forward(const std::vector<Variable>& xs) override {
		(void)xs;
		throw std::runtime_error("Model has no forward"); 
	}
	
	void plot(std::vector<Variable> inputs, std::string to_file="model");

};



class MLP : public Layer {
private:
	std::shared_ptr<Function> activation;
	std::vector<std::shared_ptr<Layer>> layers;

public:
	MLP(const std::vector<size_t>& fc_output_sizes, 
		const std::shared_ptr<Function> activation
		= std::make_shared<Sigmoid>()) : activation(activation) {
	size_t i = 0;
	for (size_t out_size : fc_output_sizes) {
		std::shared_ptr<Layer> layer = std::make_shared<Linear>(out_size);
		register_sublayers("l" + std::to_string(i), layer);
		Linear l(out_size);
		this->layers.push_back(layer);
		i++;
	}
}

	MLP(const std::initializer_list<size_t>& fc_output_sizes,
		const std::shared_ptr<Function> activation
		 = std::make_shared<Sigmoid>()) : MLP(std::vector<size_t>(fc_output_sizes), activation) {};

	Variable forward(const std::vector<Variable>& xs) override;
};


