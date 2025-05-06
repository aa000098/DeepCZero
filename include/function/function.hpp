#pragma once

#include "container/variable_all.hpp"
#include "container/tensor/tensor.hpp"

#include <vector>
#include <memory>


class Function : public std::enable_shared_from_this<Function> {
protected:
	std::vector<std::shared_ptr<VariableImpl<>>> inputs;
	std::weak_ptr<VariableImpl<>> output;
	
public:
	virtual Variable operator()(const std::vector<Variable>& inputs);

	virtual Variable forward(const std::vector<Variable>& xs) = 0;
	virtual std::vector<Variable> backward(const Variable& gy) = 0; 
	virtual std::string name();
	virtual std::uintptr_t id() {
		return reinterpret_cast<std::uintptr_t>(this);
	}

	virtual ~Function() = default;


public:
	std::vector<std::shared_ptr<VariableImpl<>>> get_inputs() { return inputs; };
	std::shared_ptr<VariableImpl<>> get_output() { return output.lock(); };

};

class Square: public Function {
public:
	Variable forward(const std::vector<Variable>& xs) override;
	std::vector<Variable> backward(const Variable& gy) override;
	~Square() = default;
};

class Exp: public Function {
public:
    Variable forward(const std::vector<Variable>& xs) override;
    std::vector<Variable> backward(const Variable& gy) override;
    ~Exp() = default;
};

class Add: public Function {
public:
    Variable forward(const std::vector<Variable>& xs) override;
    std::vector<Variable> backward(const Variable& gy) override;
    ~Add() = default;
};

class Mul: public Function {
public:
    Variable forward(const std::vector<Variable>& xs) override;
    std::vector<Variable> backward(const Variable& gy) override;
    ~Mul() = default;
};

class Neg: public Function {
public:
    Variable forward(const std::vector<Variable>& xs) override;
    std::vector<Variable> backward(const Variable& gy) override;
    ~Neg() = default;
};

class Sub: public Function {
public:
    Variable forward(const std::vector<Variable>& xs) override;
    std::vector<Variable> backward(const Variable& gy) override;
    ~Sub() = default;
};

class Div: public Function {
public:
    Variable forward(const std::vector<Variable>& xs) override;
    std::vector<Variable> backward(const Variable& gy) override;
    ~Div() = default;
};

class Pow: public Function {
public:
    Variable forward(const std::vector<Variable>& xs) override;
    std::vector<Variable> backward(const Variable& gy) override;
    ~Pow() = default;
};

class Sin: public Function {
public:
    Variable forward(const std::vector<Variable>& xs) override;
    std::vector<Variable> backward(const Variable& gy) override;
    ~Sin() = default;
};

class Cos: public Function {
public:
    Variable forward(const std::vector<Variable>& xs) override;
    std::vector<Variable> backward(const Variable& gy) override;
    ~Cos() = default;
};


