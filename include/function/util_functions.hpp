#pragma once

#include "function/function.hpp"

class Sum: public Function {
private:
	std::vector<size_t> x_shape;
	std::vector<int> axis;
	bool keepdims;

public:
	Sum(const std::vector<int> axis = {}, const bool keepdims = false) : axis(axis), keepdims(keepdims) {};

public:
	Variable forward(const std::vector<Variable>& xs) override;
	std::vector<Variable> backward(const Variable& gy) override;
	~Sum() = default;
};


class Broadcast_To : public Function {
private:
    std::vector<size_t> shape;
    std::vector<size_t> x_shape;

public:
    Broadcast_To(const std::vector<size_t>& shape) : shape(shape) {};

public:
	Variable forward(const std::vector<Variable>& xs) override;
	std::vector<Variable> backward(const Variable& gy) override;
	~Broadcast_To() = default;

};


class Sum_To : public Function {
private:
	std::vector<size_t> shape;
	std::vector<size_t> x_shape;

public:
    Sum_To(const std::vector<size_t>& shape = {}) : shape(shape) {};

public:
	Variable forward(const std::vector<Variable>& xs) override;
	std::vector<Variable> backward(const Variable& gy) override;
	~Sum_To() = default;

};

