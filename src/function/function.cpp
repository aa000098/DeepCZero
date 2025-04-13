#include "function/function.hpp"
#include "container/variable.hpp"

#include <memory>
#include <cmath>


Function::~Function() = default;

Variable Function::operator()(Variable input) {
	this->input = input.get_impl();
	forward();
	output->creator = shared_from_this();
	return Variable(output);
}


void Square::forward() {
	float x = input->data;
	float result = pow(x, 2);
	std::shared_ptr<VariableImpl<float>> output = std::make_shared<VariableImpl<float>>(result);
	set_output(output);
}

float Square::backward(float gy) {
	float x = input->data;
	return 2 * x * gy;
}


void Exp::forward() {
	float x = input->data;
	float result = exp(x);
	std::shared_ptr<VariableImpl<float>> output = std::make_shared<VariableImpl<float>>(result);
	set_output(output);
}

float Exp::backward(float gy) {
	float x = input->data;
	return exp(x) * gy;
}
