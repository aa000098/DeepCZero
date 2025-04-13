#include "function/function.hpp"
#include "container/variable.hpp"

#include <memory>
#include <cmath>


Function::~Function() = default;

Variable Function::operator()(Variable input) {
	this->input = input.get_impl();
	float y = forward();

	std::shared_ptr<VariableImpl<float>> output = std::make_shared<VariableImpl<float>>(y);
	set_output(output);

	output->creator = shared_from_this();
	return Variable(output);
}


float Square::forward() {
	float x = input->data;
	float result = pow(x, 2);
	return result;
}

float Square::backward(float gy) {
	float x = input->data;
	return 2 * x * gy;
}


float Exp::forward() {
	float x = input->data;
	float result = exp(x);
	return result;
}

float Exp::backward(float gy) {
	float x = input->data;
	return exp(x) * gy;
}
