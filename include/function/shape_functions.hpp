#pragma once

#include "function/function.hpp"

namespace function {

class Reshape: public Function {
private:
	std::vector<size_t> shape;
	std::vector<size_t> x_shape;

public:
	Reshape(const std::vector<size_t>& shape) : shape(shape) {};

public:
	Variable forward(const std::vector<Variable>& xs) override;
	std::vector<Variable> backward(const Variable& gy) override;
	~Reshape() = default;
};

class Transpose : public Function {
private:
    std::vector<size_t> axes;

public:
    Transpose(const std::vector<size_t>& axes = {}) : axes(axes) {};

public:
	Variable forward(const std::vector<Variable>& xs) override;
	std::vector<Variable> backward(const Variable& gy) override;
	~Transpose() = default;

};

}
