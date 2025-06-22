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
	std::vector<int> axes;

public:
	Softmax(std::vector<int> axes) : axes(axes) {};
	Variable forward(const std::vector<Variable>& xs) override;
	std::vector<Variable> backward(const Variable& gy) override;
	~Softmax() = default;
};


class ReLU : public Function {
public:
	Variable forward(const std::vector<Variable>& xs) override;
	std::vector<Variable> backward(const Variable& gy) override;
	~ReLU() = default;
	
};


}
