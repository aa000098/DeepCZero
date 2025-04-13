#include "container/variable.hpp"
#include "function/function.hpp"
#include <iostream>
#include <vector>

Variable::Variable(float data, bool requires_grad)
    : data(data), grad(0.0), creator(nullptr), requires_grad(requires_grad) {}


void Variable::backward() {
    if (grad == 0.0f)
		// TODO: gradient random initializing
        grad = 1.0f;

    std::vector<std::shared_ptr<Function>> funcs;
    if (creator) funcs.push_back(creator);

    while (!funcs.empty()) {
		std::shared_ptr<Function> f = funcs.back();
        funcs.pop_back();

		std::shared_ptr<Variable> input = f->get_input();
		std::shared_ptr<Variable> output = f->get_output();

		float gxs = f->backward(output->get_grad());

        if (input->grad == 0.0f)
            input->grad = gxs;
        else
            input->grad += gxs;
        
		if (input->creator)
            funcs.push_back(input->creator);
    }
}

void Variable::show() const {
    std::cout << "Variable(data=" << data << ", grad=" << grad << ")\n";
}

