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


class SiLU : public Function {
public:
	Variable forward(const std::vector<Variable>& xs) override;
	std::vector<Variable> backward(const Variable& gy) override;
	~SiLU() = default;
};


class Dropout  {
private:
	float dropout_rate;
public:
	Dropout(float dropout_rate = 0.1) : dropout_rate(dropout_rate) {};
	Variable forward(const std::vector<Variable>& xs);
	Variable operator()(const Variable& x) { 
return forward({x}); };
	~Dropout() = default;
};


}
