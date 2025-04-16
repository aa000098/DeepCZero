#pragma once

#include "container/variable.hpp"

#include <vector>
#include <memory>

using Tensor = std::vector<float>;

class Variable;

class Function : public std::enable_shared_from_this<Function> {
protected:
	std::vector<std::shared_ptr<VariableImpl>> inputs;
	std::weak_ptr<VariableImpl> output;
	
public:
	virtual Variable operator()(const std::vector<Variable>& inputs);

	virtual Tensor forward(std::vector<Tensor>& xs) = 0;
	virtual std::vector<Tensor> backward(Tensor& gy) = 0; 


public:
	std::vector<std::shared_ptr<VariableImpl>> get_inputs() { return inputs; };
	std::shared_ptr<VariableImpl> get_output() { return output.lock(); };

	virtual ~Function();
};

class Square: public Function {

public:
	Tensor forward(std::vector<Tensor>& xs) override;
	std::vector<Tensor> backward(Tensor& gy) override;

	~Square() = default;
};

class Exp: public Function {

public:
	Tensor forward(std::vector<Tensor>& xs) override;
	std::vector<Tensor> backward(Tensor& gy) override;

	~Exp() = default;
};

class Add: public Function {

public:
	Tensor forward(std::vector<Tensor>& xs) override;
	std::vector<Tensor> backward(Tensor& gy) override;

	~Add() = default;
};
