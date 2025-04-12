#include "container/variable.hpp"
#include "function/function.hpp"
#include <iostream>
#include <vector>

Variable::Variable(float data, bool requires_grad)
    : data(data), grad(0.0), creator(nullptr), requires_grad(requires_grad) {}

/*
void Variable::backward() {
    if (grad == 0.0f)
		// TODO: gradient random initializing
        grad = 1.0f;

    std::vector<std::shared_ptr<Function>> funcs;
    if (creator) funcs.push_back(creator);

    while (!funcs.empty()) {
		std::shared_ptr<Function> f = funcs.back();
        funcs.pop_back();

		std::shared_ptr<Variable> &input = f.get_input();
		std::shared_ptr<Variable> &output = f.get_output();

		f.backward(output.get_grad());
		auto gxs = f.backward();

        for (size_t i = 0; i < inputs.size(); ++i) {
            if (inputs[i]->grad == 0.0f)
                inputs[i]->grad = gxs[i]->data;
            else
                inputs[i]->grad += gxs[i]->data;

            if (inputs[i]->creator)
                funcs.push_back(inputs[i]->creator);
        }
    }
}
*/
void Variable::show() const {
    std::cout << "Variable(data=" << data << ", grad=" << grad << ")\n";
}

