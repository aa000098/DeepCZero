#include "container/variable.h"
#include "container/function.h"
#include <iostream>
#include <vector>

Variable::Variable(float data, bool requires_grad)
    : data(data), grad(0.0), requires_grad(requires_grad), creator(nullptr) {}

void Variable::set_creator(std::shared_ptr<Function> func) {
    creator = func;
}

void Variable::backward() {
    if (grad == 0.0f)
        grad = 1.0f;

    std::vector<std::shared_ptr<Function>> funcs;
    if (creator) funcs.push_back(creator);

    while (!funcs.empty()) {
		auto f = funcs.back();
        funcs.pop_back();

		std::vector<std::shared_ptr<Variable>> &inputs = f->inputs;
		std::vector<std::shared_ptr<Variable>> &outputs = f->outputs;

		std::vector<float> gxs = f->backward(outputs[0]->grad);

        for (size_t i = 0; i < inputs.size(); ++i) {
            if (inputs[i]->grad == 0.0f)
                inputs[i]->grad = gxs[i];
            else
                inputs[i]->grad += gxs[i];

            if (inputs[i]->creator)
                funcs.push_back(inputs[i]->creator);
        }
    }
}

void Variable::show() const {
    std::cout << "Variable(data=" << data << ", grad=" << grad << ")\n";
}

