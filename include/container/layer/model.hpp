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



class MLP : public Model {
private:
	std::shared_ptr<Function> activation;
	std::vector<std::shared_ptr<Layer>> layers;

public:
	MLP(const std::vector<size_t>& fc_output_sizes, 
		const std::shared_ptr<Function> activation
		= std::make_shared<Sigmoid>());

	MLP(const std::initializer_list<size_t>& fc_output_sizes,
		const std::shared_ptr<Function> activation
		 = std::make_shared<Sigmoid>()) 
			: MLP(std::vector<size_t>(fc_output_sizes), 
					activation) {};

	Variable forward(const std::vector<Variable>& xs) override;
};


