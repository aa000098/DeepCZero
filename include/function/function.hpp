#pragma once

#include "container/variable.hpp"

#include <vector>
#include <memory>

using Tensor = std::vector<float>;

class Variable;

class Function : public std::enable_shared_from_this<Function> {
protected:
	std::shared_ptr<VariableImpl> input;
	std::shared_ptr<VariableImpl> output;
	
public:
	virtual Variable operator()(const Variable& input);

	virtual Tensor forward(Tensor xs) = 0;
	virtual Tensor backward(Tensor gy) = 0; 


public:
	std::shared_ptr<VariableImpl> get_input() { return input; };
	std::shared_ptr<VariableImpl> get_output() { return output; };

	virtual ~Function();
};

class Square: public Function {

public:
	Tensor forward(Tensor xs) override;
	Tensor backward(Tensor gy) override;

	~Square() = default;
};

class Exp: public Function {

public:
	Tensor forward(Tensor xs) override;
	Tensor backward(Tensor gy) override;

	~Exp() = default;
};
