#pragma once

#include <vector>
#include <memory>

class Variable;

class Function : public std::enable_shared_from_this<Function> {
protected:
	std::shared_ptr<Variable> input;
	std::shared_ptr<Variable> output;
	std::vector<std::shared_ptr<Variable>> inputs;
	std::vector<std::shared_ptr<Variable>> outputs;
	
public:
	virtual Variable operator()(Variable input);

	virtual void forward() = 0;
	virtual float backward(float gy) = 0; 
	
//	virtual void forward(std::vector<std::shared_ptr<Variable>> inputs);
	
//	virtual void backward(std::vector<std::shared_ptr<Variable>> grad_output);

public:
	const std::shared_ptr<Variable> &get_input() { return input;} ;
	const std::shared_ptr<Variable> &get_output() { return output; };
	void set_output(std::shared_ptr<Variable> output) { this->output = output; };

	virtual ~Function();


};

class Square: public Function {

public:
	void forward() override;
	float backward(float gy) override;
	
//	void forward(std::vector<std::shared_ptr<Variable>> inputs);
	
//	void backward(std::shared_ptr<Variable> grad_output);
//	void backward(std::vector<std::shared_ptr<Variable>> grad_output);

	~Square() = default;
};

class Exp: public Function {

public:
	void forward() override;
	float backward(float gy) override;

};
