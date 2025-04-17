#include "container/variable.hpp"
#include "function/function.hpp"
#include "graph/graph.hpp"

#include <unordered_set>
#include <iostream>

Variable::Variable(const Tensor& data, bool requires_grad) : impl(std::make_shared<VariableImpl>(data, requires_grad)) {}

Variable::Variable(std::shared_ptr<VariableImpl> impl) : impl(std::move(impl)) {}

void Variable::backward(bool retain_grad) {
	impl->grad = Tensor(impl->data.size(), 1.0f);

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

		if (!retain_grad) output->grad = Tensor({0});
    }
}

void Variable::show() const {
    std::cout << "Variable {\n";
	std::cout << "  data: [ ";
	for (size_t i = 0; i < impl->data.size(); ++i) {
		std::cout << impl->data[i];
		if (i != impl->data.size() -1) std::cout << ", ";
	}
	std:: cout << " ]\n";

    std::cout << "  grad: [ ";
    for (size_t i = 0; i < impl->grad.size(); ++i) {
        std::cout << impl->grad[i];
        if (i != impl->grad.size() - 1) std::cout << ", ";
    }
    std::cout << " ]\n";

	std::cout << "}" << std::endl;
}
