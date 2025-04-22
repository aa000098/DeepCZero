#include "container/variable.hpp"
#include "function/function.hpp"
#include "graph/graph.hpp"

#include <unordered_set>
#include <string>
#include <iostream>

VariableImpl::VariableImpl(
		const std::vector<size_t>& shape,
		const std::vector<float>& vec,
		std::string name, 
		bool requires_grad) 
		: name(name), grad(shape, {}), creator(), requires_grad(requires_grad) {
	if (shape.size() == 1)
		data = tensor::Tensor1D(vec);
	else
		data = tensor::TensorND(shape, vec);
	grad = tensor::TensorND(shape, {});
};

void Variable::backward(bool retain_grad) {

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
					input->grad.raw_data()[j] = input->grad.raw_data()[j] + gx.raw_data()[j];
			}
		}

		if (!retain_grad) output->grad = Tensor(output->data.get_shape(), 0.0f);
    }
}

void print_tensor(const Tensor& tensor, size_t depth = 0, size_t offset = 0) {
	const std::vector<size_t> shape = tensor.get_shape();
    if (tensor.ndim()-1 == depth) {
        std::cout << "[ ";
        for (size_t i = offset*shape[depth]; i < (offset+1)*shape[depth]; ++i) {
            std::cout << tensor.raw_data()[i];
            if (i != (offset+1)*shape[depth] - 1) std::cout << ", ";
        }
        std::cout << " ]";
    } else {
        std::vector<size_t> shape = tensor.get_shape();
        size_t dim = shape[depth];
        std::cout << "[";
        for (size_t i = 0; i < dim; ++i) {
            if (i > 0) std::cout << " ";
            print_tensor(tensor, depth + 1, offset+i);
            if (i != dim - 1) std::cout << "," << std::endl;
        }
        std::cout << "]";
    }
}

void Variable::show() const {
    std::cout << "Variable {\n";
    std::cout << "  data: \n";
    print_tensor(impl->data);
    std::cout << std::endl;

    std::cout << "  name: " << impl->name << std::endl;

    std::cout << "  grad: \n";
    print_tensor(impl->grad);
    std::cout << std::endl;

    std::cout << "}" << std::endl;
}
