#include "container/layer/model.hpp"
#include "graph/utils/utils.hpp"

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
