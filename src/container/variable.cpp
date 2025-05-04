#include "container/variable.hpp"
#include "container/tensor/tensor.hpp"
#include "function/function.hpp"
#include "graph/graph.hpp"

#include <unordered_set>
#include <string>
#include <iostream>

void Variable::backward(bool retain_grad) {
	impl->grad = Tensor<>(impl->data.get_shape(), 1);
	auto creator = impl->creator.get();
	if (!creator) return;
	Graph graph(creator);
	std::vector<Function*> topo_order = graph.get_topo_order();

	for (auto& f : topo_order) {
		std::vector<std::shared_ptr<VariableImpl<>>> inputs = f->get_inputs();
		std::shared_ptr<VariableImpl<>> output = f->get_output();

		Tensor gy = output->grad;
		std::vector<Tensor<>> gxs = f->backward(gy);
		for (size_t i = 0; i < gxs.size(); ++i) {
			std::shared_ptr<VariableImpl<>> input = inputs[i];
			const Tensor<>& gx = gxs[i];
			if (input->grad.empty())
				input->grad = gx;
			else {
				for (size_t j = 0; j < input->grad.size(); ++j)
					input->grad.raw_data()[j] = input->grad.raw_data()[j] + gx.raw_data()[j];
			}
		}
		if (!retain_grad) output->grad = Tensor<>();
	}
}

void Variable::show() const {
	std::cout << "Variable {\n";
	std::cout << "  data: \n";
	impl->data.show();
	std::cout << "  name: " << impl->name << std::endl;
	std::cout << "  grad: \n";
	impl->grad.show();
	std::cout << "}\n";
}
