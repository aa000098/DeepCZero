#include "function/function.hpp"
#include "container/variable.hpp"

#include <iostream>
#include <vector>
#include <memory>
#include <cmath>


Function::~Function() = default;

Variable Function::operator()(Variable input) {
	this->input = std::make_shared<Variable>(input);
	forward();
	return *get_output();
}


void Square::forward() {
	float x = input->get_data();
	float result = pow(x, 2);
	std::shared_ptr<Variable> output = std::make_shared<Variable>(result);
	output->set_creator(shared_from_this());
	set_output(output);
}

float Square::backward(float gy) {
	float x = input->get_data();
	return 2 * x * gy;
}


void Exp::forward() {
	float x = input->get_data();
	float result = exp(x);
	std::shared_ptr<Variable> output = std::make_shared<Variable>(result);
	output->set_creator(shared_from_this());
	set_output(output);
}

float Exp::backward(float gy) {
	float x = input->get_data();
	return exp(x) * gy;
}
