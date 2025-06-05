#include "container/layer/model.hpp"
#include "graph/utils/utils.hpp"

void Model::plot(std::vector<Variable> inputs,
				std::string to_file) {

	Variable y = this->forward(inputs);
	plot_dot_graph(y, true, to_file);
}


Variable MLP::forward(const std::vector<Variable>& xs) {
	Variable x = xs[0];

	for (size_t i = 0; i < layers.size() - 1; i++) {
	x = (*layers[i])(x);
		x = (*activation)(x);
	}
	return (*layers.back())(x);
}
