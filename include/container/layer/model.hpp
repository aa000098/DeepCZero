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
	const std::string WEIGHTS_PATH = "https://github.com/koki0702/dezero-models/releases/download/v0.1/vgg16.npz";
		
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

	void load_weights(const std::string& wegiths_path);
	Variable forward(const std::vector<Variable>& xs) override;

	// Getter for sublayers
	std::shared_ptr<Layer> get_layer(const std::string& name) const {
		if (name == "conv1_1") return conv1_1;
		if (name == "conv1_2") return conv1_2;
		if (name == "conv2_1") return conv2_1;
		if (name == "conv2_2") return conv2_2;
		if (name == "conv3_1") return conv3_1;
		if (name == "conv3_2") return conv3_2;
		if (name == "conv3_3") return conv3_3;
		if (name == "conv4_1") return conv4_1;
		if (name == "conv4_2") return conv4_2;
		if (name == "conv4_3") return conv4_3;
		if (name == "conv5_1") return conv5_1;
		if (name == "conv5_2") return conv5_2;
		if (name == "conv5_3") return conv5_3;
		if (name == "fc6") return fc6;
		if (name == "fc7") return fc7;
		if (name == "fc8") return fc8;
		return nullptr;
	}
};

class SimpleRNN : public Model {
private:
	std::shared_ptr<Layer> rnn;
	std::shared_ptr<Layer> fc;

public:
	SimpleRNN(size_t hidden_size, size_t output_size);
	void reset_state();
	Variable forward(const std::vector<Variable>& xs) override;
};