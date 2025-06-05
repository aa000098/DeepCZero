#include "container/layer/model.hpp"
#include "graph/utils/utils.hpp"

void Model::plot(std::vector<Variable> inputs,
				std::string to_file) {

	Variable y = this->forward(inputs);
	plot_dot_graph(y, true, to_file);
}
