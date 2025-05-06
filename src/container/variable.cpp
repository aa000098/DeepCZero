#include "container/variable_all.hpp"
#include "container/tensor/tensor_all.hpp"
#include "function/function.hpp"
#include "graph/graph.hpp"

#include <unordered_set>
#include <string>
#include <iostream>

void Variable::backward(bool retain_grad) {
	std::shared_ptr<Variable> grad = std::make_shared<Variable>(Tensor<>(impl->data.get_shape(), 1));
	impl->grad = grad;
	auto creator = impl->creator.get();
	if (!creator) return;
	Graph graph(creator);
	std::vector<Function*> topo_order = graph.get_topo_order();

	for (auto& f : topo_order) {
		std::vector<std::shared_ptr<VariableImpl<>>> inputs = f->get_inputs();
		std::shared_ptr<VariableImpl<>> output = f->get_output();

		std::shared_ptr<Variable> gy = output->grad;
		std::vector<Variable> gxs = f->backward(*gy);
		for (size_t i = 0; i < gxs.size(); ++i) {
			std::shared_ptr<VariableImpl<>> input = inputs[i];
			const Variable& gx = gxs[i];
			if (!input->grad)
				input->grad = std::make_shared<Variable>(gx);
			else 
				(*input->grad) += gx;
		}
		if (!retain_grad) output->grad.reset();
	}
}

void Variable::show() const {
	std::cout << "Variable {\n";

	const auto& data = impl->data;
	auto shape = data.get_shape();
	std::cout << "  data: ";
	if (shape.size() == 1) data.show();
	else {
		std::cout << "\n";
		data.show();
	}
	std::cout << "  name: " << (impl->name.empty() ? "(unnamed)" : impl->name) << std::endl;
	
	std::cout << "  grad: ";
	if (impl->grad) {
		auto gshape = impl->grad->data().get_shape();
		if (gshape.size() == 1) impl->grad->data().show();
		else {
			std::cout << "\n";
			impl->grad->data().show();
		}
	} else
		std::cout << "(no grad)\n";
	
	std::cout << "}\n";
}
