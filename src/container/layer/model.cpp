#include "container/layer/model.hpp"
#include "graph/utils/utils.hpp"
#include "utils/io.hpp"
#include "cnpy.h"

#include <iostream>

void Model::plot(std::vector<Variable> inputs,
				std::string to_file) {

	Variable y = this->forward(inputs);
	plot_dot_graph(y, true, to_file);
}

MLP::MLP(const std::vector<size_t>& fc_output_sizes,
		const std::shared_ptr<Function> activation) 
			: activation(activation) {
	size_t i = 0;
	for (size_t out_size : fc_output_sizes) {
		std::shared_ptr<Layer> layer = std::make_shared<Linear>(out_size);
		register_sublayers("l" + std::to_string(i), layer);
		Linear l(out_size);
		this->layers.push_back(layer);
		i++;
	}
}

Variable MLP::forward(const std::vector<Variable>& xs) {
	Variable x = xs[0];

	for (size_t i = 0; i < layers.size() - 1; i++) {
	x = (*layers[i])(x);
		x = (*activation)(x);
	}
	return (*layers.back())(x);
}

VGG16::VGG16(bool pretrained) {
	std::pair<size_t, size_t> k = {3, 3};
	std::pair<size_t, size_t> s = {1, 1};
	std::pair<size_t, size_t> p = {1, 1};
	conv1_1 = std::make_shared<Conv2d>(64, k, s, p);
	conv1_2 = std::make_shared<Conv2d>(64, k, s, p);
	conv2_1 = std::make_shared<Conv2d>(128, k, s, p);
	conv2_2 = std::make_shared<Conv2d>(128, k, s, p);
	conv3_1 = std::make_shared<Conv2d>(256, k, s, p);
	conv3_2 = std::make_shared<Conv2d>(256, k, s, p);
	conv3_3 = std::make_shared<Conv2d>(256, k, s, p);
	conv4_1 = std::make_shared<Conv2d>(512, k, s, p);
	conv4_2 = std::make_shared<Conv2d>(512, k, s, p);
	conv4_3 = std::make_shared<Conv2d>(512, k, s, p);
	conv5_1 = std::make_shared<Conv2d>(512, k, s, p);
	conv5_2 = std::make_shared<Conv2d>(512, k, s, p);
	conv5_3 = std::make_shared<Conv2d>(512, k, s, p);
	fc6 = std::make_shared<Linear>(4096);
	fc7 = std::make_shared<Linear>(4096);
	fc8 = std::make_shared<Linear>(1000);

	register_sublayers("conv1_1", conv1_1);
	register_sublayers("conv1_2", conv1_2);
	register_sublayers("conv2_1", conv2_1);
	register_sublayers("conv2_2", conv2_2);
	register_sublayers("conv3_1", conv3_1);
	register_sublayers("conv3_2", conv3_2);
	register_sublayers("conv3_3", conv3_3);
	register_sublayers("conv4_1", conv4_1);
	register_sublayers("conv4_2", conv4_2);
	register_sublayers("conv4_3", conv4_3);
	register_sublayers("conv5_1", conv5_1);
	register_sublayers("conv5_2", conv5_2);
	register_sublayers("conv5_3", conv5_3);
	register_sublayers("fc6", fc6);
	register_sublayers("fc7", fc7);
	register_sublayers("fc8", fc8);
   
	if (pretrained) {
		std::string weights_path = get_file(this->WEIGHTS_PATH, "weights/vgg16.npz");
		this->load_weights(weights_path);
	}
}

void VGG16::load_weights(const std::string& weights_path) {
	// std::cerr << "[VGG16] Loading weights from: " << weights_path << std::endl;

	cnpy::npz_t npz = cnpy::npz_load(weights_path);

	// Conv layers
	conv1_1->load_params_from_npz(npz, "conv1_1");
	conv1_2->load_params_from_npz(npz, "conv1_2");
	conv2_1->load_params_from_npz(npz, "conv2_1");
	conv2_2->load_params_from_npz(npz, "conv2_2");
	conv3_1->load_params_from_npz(npz, "conv3_1");
	conv3_2->load_params_from_npz(npz, "conv3_2");
	conv3_3->load_params_from_npz(npz, "conv3_3");
	conv4_1->load_params_from_npz(npz, "conv4_1");
	conv4_2->load_params_from_npz(npz, "conv4_2");
	conv4_3->load_params_from_npz(npz, "conv4_3");
	conv5_1->load_params_from_npz(npz, "conv5_1");
	conv5_2->load_params_from_npz(npz, "conv5_2");
	conv5_3->load_params_from_npz(npz, "conv5_3");

	// FC layers
	fc6->load_params_from_npz(npz, "fc6");
	fc7->load_params_from_npz(npz, "fc7");
	fc8->load_params_from_npz(npz, "fc8");

	// std::cerr << "[VGG16] Weights loaded successfully" << std::endl;
}
             
Variable VGG16::forward(const std::vector<Variable>& xs) {

	// [B, C_in, H, W]
	Variable x = xs[0];
	
	// [B, 64, H, W]
	x =  relu((*conv1_1)(x));
	x =  relu((*conv1_2)(x));
	// [B, 64, H/2, W/2]
	x = pooling(x, {2, 2}, {2, 2});

	// [B, 128, H/2, W/2]
	x =  relu((*conv2_1)(x));
	x =  relu((*conv2_2)(x));
	// [B, 128, H/4, W/4]
	x = pooling(x, {2, 2}, {2, 2});

	// [B, 256, H/4, W/4]
	x =  relu((*conv3_1)(x));
	x =  relu((*conv3_2)(x));
	x =  relu((*conv3_3)(x));
	// [B, 256, H/8, W/8]
	x = pooling(x, {2, 2}, {2, 2});

	// [B, 512, H/8, W/8]
	x =  relu((*conv4_1)(x));
	x =  relu((*conv4_2)(x));
	x =  relu((*conv4_3)(x));
	// [B, 512, H/16, W/16]
	x = pooling(x, {2, 2}, {2, 2});

	// [B, 512, H/16, W/16]
	x =  relu((*conv5_1)(x));
	x =  relu((*conv5_2)(x));
	x =  relu((*conv5_3)(x));
	// [B, 512, H/32, W/32]
	x = pooling(x, {2, 2}, {2, 2});

	std::vector<size_t> x_shape = x.shape();
	size_t flatten_dim = x_shape[1] * x_shape[2] * x_shape[3]; 
	// [B, 512 * H/32 * W/32]
	x = reshape(x, {x.shape()[0], flatten_dim});

	// [B, 4096]
	x = dropout(relu((*fc6)(x)));
	x = dropout(relu((*fc7)(x)));

	// [B, 1000]
	x = (*fc8)(x);

	return x;
}

SimpleRNN::SimpleRNN(size_t hidden_size, size_t output_size) {
	rnn = std::make_shared<layer::RNN>(hidden_size);
	fc = std::make_shared<layer::Linear>(output_size);

	register_sublayers("rnn", rnn);
	register_sublayers("fc", fc);
}

void SimpleRNN::reset_state() {
	std::dynamic_pointer_cast<layer::RNN>(rnn)->reset_state();
}

Variable SimpleRNN::forward(const std::vector<Variable>& xs) {
	Variable h = (*rnn)({xs[0]});
	Variable y = (*fc)({h});
	return y;
}