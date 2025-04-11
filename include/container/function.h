#pragma once

#include <vector>
#include <memory>

class Variable;

class Function {

public:
	std::vector<std::shared_ptr<Variable>> inputs;
	std::vector<std::shared_ptr<Variable>> outputs;

	virtual std::shared_ptr<Variable> operator()(std::shared_ptr<Variable> input) = 0;
	virtual std::vector<float> backward(float grad_output) = 0;

	virtual ~Function() = default;


};
