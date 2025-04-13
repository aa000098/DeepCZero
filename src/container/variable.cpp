#include "container/variable.hpp"
#include "function/function.hpp"
#include <iostream>

Variable::Variable(const Tensor& data, bool requires_grad) : impl(std::make_shared<VariableImpl>(data, requires_grad)) {}

Variable::Variable(std::shared_ptr<VariableImpl> impl) : impl(std::move(impl)) {}

void Variable::backward() {
    impl->grad = Tensor(impl->data.size(), 1.0f);

	std::vector<std::shared_ptr<Function>> funcs;
	if (impl->creator) funcs.push_back(impl->creator);

    while (!funcs.empty()) {
		std::shared_ptr<Function> f = funcs.back();
        funcs.pop_back();

		std::shared_ptr<VariableImpl> input = f->get_input();
		std::shared_ptr<VariableImpl> output = f->get_output();

		Tensor gys = output->grad;
		Tensor gxs = f->backward(gys);
        if (input->grad.empty())
            input->grad = gxs;
        else {
            for (size_t i = 0; i < input->grad.size(); ++i)
				input->grad[i] += gxs[i];
		}
        
		if (input->creator)
            funcs.push_back(input->creator);
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
