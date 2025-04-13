#pragma once

#include "container/variable.hpp"

#include <vector>
#include <memory>

class Variable;

class Function : public std::enable_shared_from_this<Function> {
protected:
	std::shared_ptr<VariableImpl<float>> input;
	std::shared_ptr<VariableImpl<float>> output;
//	std::vector<std::shared_ptr<Variable>> inputs;
//	std::vector<std::shared_ptr<Variable>> outputs;
	
public:
	virtual Variable operator()(Variable input);

	virtual float forward() = 0;
	virtual float backward(float gy) = 0; 
	
//	virtual void forward(std::vector<std::shared_ptr<Variable>> inputs);
	
//	virtual void backward(std::vector<std::shared_ptr<Variable>> grad_output);

public:
	const std::shared_ptr<VariableImpl<float>> &get_input() { return input;} ;
	const std::shared_ptr<VariableImpl<float>> &get_output() { return output; };
	void set_output(std::shared_ptr<VariableImpl<float>> output) { this->output = output; };

	virtual ~Function();


};

class Square: public Function {

public:
	float forward() override;
	float backward(float gy) override;
	
	~Square() = default;
};

class Exp: public Function {

public:
	float forward() override;
	float backward(float gy) override;

	~Exp() = default;
};
