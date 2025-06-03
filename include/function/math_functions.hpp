#pragma once

#include "function/function.hpp"

namespace function {

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
private:
	std::vector<size_t> x0_shape;
	std::vector<size_t> x1_shape;

public:
    Variable forward(const std::vector<Variable>& xs) override;
    std::vector<Variable> backward(const Variable& gy) override;
    ~Add() = default;
};

class Mul: public Function {
private:
	std::vector<size_t> x0_shape;
	std::vector<size_t> x1_shape;

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
private:
	std::vector<size_t> x0_shape;
	std::vector<size_t> x1_shape;

public:
    Variable forward(const std::vector<Variable>& xs) override;
    std::vector<Variable> backward(const Variable& gy) override;
    ~Sub() = default;
};

class Div: public Function {
private:
	std::vector<size_t> x0_shape;
	std::vector<size_t> x1_shape;

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

class Tanh: public Function {
public:
    Variable forward(const std::vector<Variable>& xs) override;
    std::vector<Variable> backward(const Variable& gy) override;
    ~Tanh() = default;
};

class MatMul: public Function {
public:
    Variable forward(const std::vector<Variable>& xs) override;
    std::vector<Variable> backward(const Variable& gy) override;
    ~MatMul() = default;
};

class Linear: public Function {
public:
    Variable forward(const std::vector<Variable>& xs) override;
    std::vector<Variable> backward(const Variable& gy) override;
    ~Linear() = default;
};

}
