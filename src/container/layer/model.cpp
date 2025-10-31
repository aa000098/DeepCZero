#include "container/layer/model.hpp"
#include "graph/utils/utils.hpp"
#include "utils/io.hpp"

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
   
	if (pretrained) {
		std::string weights_path = get_file(this->WEIGHTS_PATH);
	}
}
        
Variable VGG16::forward(const std::vector<Variable>& xs) {
	Variable x = xs[0];
	std::cout << "1" << std::endl;
	x =  relu((*conv1_1)(x));
	std::cout << "2" << std::endl;

	x =  relu((*conv1_2)(x));
	x = pooling(x, {2, 2}, {2, 2});
	x =  relu((*conv2_1)(x));
	x =  relu((*conv2_2)(x));
	x = pooling(x, {2, 2}, {2, 2});
	x =  relu((*conv3_1)(x));
	x =  relu((*conv3_2)(x));
	x =  relu((*conv3_3)(x));
	x = pooling(x, {2, 2}, {2, 2});
	x =  relu((*conv4_1)(x));
	x =  relu((*conv4_2)(x));
	x =  relu((*conv4_3)(x));
	x = pooling(x, {2, 2}, {2, 2});
	x =  relu((*conv5_1)(x));
	x =  relu((*conv5_2)(x));
	x =  relu((*conv5_3)(x));
	x = pooling(x, {2, 2}, {2, 2});
	x = reshape(x, {x.shape()[0], x.shape()[3]});
	x = dropout(relu((*fc6)(x)));
	x = dropout(relu((*fc7)(x)));
	x = (*fc8)(x);
	return x;
}
