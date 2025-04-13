#include "function/function.hpp"
#include "container/variable.hpp"

#include <memory>
#include <cmath>


Function::~Function() = default;

Variable Function::operator()(const Variable& input) {
	this->input = input.get_impl();
	Tensor xs = this->input->data;
	
	Tensor ys = forward(xs);
	
	output = std::make_shared<VariableImpl>(ys);
	
	output->creator = shared_from_this();

	return Variable(output);
}


Tensor Square::forward(Tensor xs) {
	Tensor results;
	for (float x : xs) {
		float y = pow(x, 2);
		results.push_back(y);
	}
	return results;
}

Tensor Square::backward(Tensor gy) {
	Tensor xs = input->data;
	Tensor results;
	for (size_t i = 0; i < xs.size(); ++i) {
		float gx = 2 * xs[i] * gy[i];
		results.push_back(gx);
	}
	return results;
}


Tensor Exp::forward(Tensor xs) {
	Tensor results;
	for (float x : xs) {
		float y = exp(x);
		results.push_back(y);
	}
	return results;
}

Tensor Exp::backward(Tensor gy) {
	Tensor xs = input->data;
	Tensor results;
	for (size_t i = 0; i < xs.size(); ++i) {
		float gx = exp(xs[i]) * gy[i];
		results.push_back(gx);
	}
	return results;
}

