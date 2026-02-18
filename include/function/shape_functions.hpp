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

class Concat : public Function {
private:
	int axis;
	std::vector<size_t> split_sizes;

public:
	Concat(int axis = 1) : axis(axis) {};
	Variable forward(const std::vector<Variable>& xs) override;
	std::vector<Variable> backward(const Variable& gy) override;
	~Concat() = default;
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

class Upsample : public Function {
private:
	size_t scale_factor;
	std::vector<size_t> input_shape;

public:
	Upsample(size_t scale_factor = 2) : scale_factor(scale_factor) {};
	Variable forward(const std::vector<Variable>& xs) override;
	std::vector<Variable> backward(const Variable& gy) override;
	~Upsample() = default;
};

}
