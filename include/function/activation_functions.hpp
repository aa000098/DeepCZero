#pragma once

#include "function/function.hpp"

namespace function {
class Sigmoid : public Function {

public:
	Variable forward(const std::vector<Variable>& xs) override;
	std::vector<Variable> backward(const Variable& gy) override;
	~Sigmoid() = default;
};

class Softmax : public Function {
private:
	std::vector<int> axis;

public:
	Softmax(std::vector<int> axis) : axis(axis) {};
	Variable forward(const std::vector<Variable>& xs) override;
	std::vector<Variable> backward(const Variable& gy) override;
	~Softmax() = default;
};

}
