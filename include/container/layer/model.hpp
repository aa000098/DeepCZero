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
using layer::Conv2d;

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

class VGG16 : public Model {
private:
	std::string WEIGHTS_PATH = "https://github.com/koki0702/dezero-models/releases/download/v0.1/vgg16.npz";
		
private:
	std::shared_ptr<Layer> conv1_1;
	std::shared_ptr<Layer> conv1_2;
	std::shared_ptr<Layer> conv2_1;
	std::shared_ptr<Layer> conv2_2;
	std::shared_ptr<Layer> conv3_1;
	std::shared_ptr<Layer> conv3_2;
	std::shared_ptr<Layer> conv3_3;
	std::shared_ptr<Layer> conv4_1;
	std::shared_ptr<Layer> conv4_2;
	std::shared_ptr<Layer> conv4_3;
	std::shared_ptr<Layer> conv5_1;
	std::shared_ptr<Layer> conv5_2;
	std::shared_ptr<Layer> conv5_3;
	std::shared_ptr<Layer> fc6;
	std::shared_ptr<Layer> fc7;
	std::shared_ptr<Layer> fc8;

public:
	VGG16(bool pretrained = false);

	Variable forward(const std::vector<Variable>& xs) override;
};
