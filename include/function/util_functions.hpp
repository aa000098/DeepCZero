#pragma once

#include "function/function.hpp"

class Sum: public Function {
private:
	std::vector<size_t> x_shape;
	std::vector<size_t> axis;
	bool keepdims;

public:
	Sum(const std::vector<size_t> axis = {}, const bool keepdims = false) : axis(axis), keepdims(keepdims) {};

public:
	Variable forward(const std::vector<Variable>& xs) override;
	std::vector<Variable> backward(const Variable& gy) override;
	~Sum() = default;
};

class Sum_To : public Function {
private:
    std::vector<size_t> axes;

public:
    Sum_To(const std::vector<size_t>& axes = {}) : axes(axes) {};

public:
	Variable forward(const std::vector<Variable>& xs) override;
	std::vector<Variable> backward(const Variable& gy) override;
	~Sum_To() = default;

};

