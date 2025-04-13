#include "container/variable.hpp"
#include "function/function.hpp"
#include <iostream>
#include <vector>

VariableImpl::VariableImpl(float data, bool requires_grad)
    : data(data), grad(0.0), creator(nullptr), requires_grad(requires_grad) {}

Variable::Variable(float data, bool requires_grad) : impl(std::make_shared<VariableImpl>(data, requires_grad)) {}

Variable::Variable(std::shared_ptr<VariableImpl> impl) : impl(impl) {}

void Variable::backward() {
    if (get_grad() == 0.0f)
		// TODO: gradient random initializing
        set_grad(1.0f);

    std::vector<std::shared_ptr<Function>> funcs;
    if (get_creator()) funcs.push_back(get_creator());

    while (!funcs.empty()) {
		std::shared_ptr<Function> f = funcs.back();
        funcs.pop_back();

		std::shared_ptr<VariableImpl> input = f->get_input();
		std::shared_ptr<VariableImpl> output = f->get_output();

		float gxs = f->backward(output->grad);
		
        if (input->grad == 0.0f)
            input->grad = gxs;
        else
            input->grad = gxs;
        
		if (input->creator)
            funcs.push_back(input->creator);
    }
}

void Variable::show() const {
    std::cout << "Variable(data=" << get_data() << ", grad=" << get_grad() << ")\n";
}

