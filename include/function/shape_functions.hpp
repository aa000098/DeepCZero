#pragma once

#include "function/function.hpp"

class Reshape: public Function {
private:
	std::vector<size_t> shape;
	std::vector<size_t> x_shape;

public:
	Reshape(std::vector<size_t> shape) : shape(shape) {};

public:
	Variable forward(const std::vector<Variable>& xs) override;
	std::vector<Variable> backward(const Variable& gy) override;
	~Reshape() = default;
};

