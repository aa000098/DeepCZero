#include "function/function.hpp"
#include "container/variable.hpp"

#include <iostream>
#include <vector>
#include <memory>
#include <cmath>


Function::~Function() = default;


void Square::forward(std::shared_ptr<Variable> input) {
	float result = pow(input->get_data(), 2);
	std::shared_ptr<Variable> output = std::make_shared<Variable>(result);
	set_output(output);
}

Variable Square::operator()(Variable input) {
	std::shared_ptr<Variable> shared_input = std::make_shared<Variable>(input);
	forward(shared_input);
	return *get_output();
}


void Exp::forward(std::shared_ptr<Variable> input) {
	float result = exp(input->get_data());
	std::shared_ptr<Variable> output = std::make_shared<Variable>(result);
	set_output(output);
}

Variable Exp::operator()(Variable input) {
	std::shared_ptr<Variable> shared_input = std::make_shared<Variable>(input);
	forward(shared_input);
	return *get_output();
}

