#include "container/variable.hpp"
#include "function/function.hpp"
#include "graph/graph.hpp"

#include <unordered_set>
#include <string>
#include <iostream>

void Variable::backward(bool retain_grad) {
	impl->grad = Tensor(impl->data.get_shape(), 1.0f);

	auto creator = impl->creator.get();
	if (!creator) return;
	Graph graph;
	graph.build_from(creator);
	std::vector<Function*> topo_order = graph.get_topo_order();

	for (auto& f : topo_order) {
		std::vector<std::shared_ptr<VariableImpl>> inputs = f->get_inputs();
		std::shared_ptr<VariableImpl> output = f->get_output();

		Tensor gy = output->grad;
		std::vector<Tensor> gxs = f->backward(gy);

		for (size_t i = 0; i < inputs.size(); ++i) {
			std::shared_ptr<VariableImpl> input = inputs[i];
			const Tensor& gx = gxs[i];

			if (input->grad.empty())
            	input->grad = gx;
        	else {
            	for (size_t j = 0; j < input->grad.size(); ++j)
					input->grad[j] += gx[j];
			}
		}

		if (!retain_grad) output->grad = Tensor(output->data.get_shape(), 0.0f);
    }
}

void print_tensor(const Tensor& tensor, size_t depth = 0, std::vector<size_t> prefix = {}) {
    if (tensor.ndim() == 1) {
        std::cout << "[ ";
        for (size_t i = 0; i < tensor.size(); ++i) {
            std::cout << tensor[i];
            if (i != tensor.size() - 1) std::cout << ", ";
        }
        std::cout << " ]";
    } else {
        std::vector<size_t> shape = tensor.get_shape();
        size_t dim = shape[depth];
        std::cout << "[";
        for (size_t i = 0; i < dim; ++i) {
            std::vector<size_t> new_prefix = prefix;
            new_prefix.push_back(i);
            if (i > 0) std::cout << " ";
            print_tensor(tensor, depth + 1, new_prefix);
            if (i != dim - 1) std::cout << "," << std::endl;
        }
        std::cout << "]";
    }
}

void Variable::show() const {
    std::cout << "Variable {\n";
    std::cout << "  data: ";
    print_tensor(impl->data);
    std::cout << std::endl;

    std::cout << "  name: " << impl->name << std::endl;

    std::cout << "  grad: ";
    print_tensor(impl->grad);
    std::cout << std::endl;

    std::cout << "}" << std::endl;
}
