#include "function/function.hpp"
#include "config/config.hpp"
#include "container/tensor/tensor_all.hpp"

#include <cmath>
#include <algorithm>
#include <memory>


Variable Function::operator()(const std::vector<Variable>& inputs) {
	this->inputs.clear();

	if (dcz::Config::get().enable_backprop) {
		for (const auto& input : inputs) {
			std::shared_ptr<VariableImpl<>> impl = input.get_impl();
			this->inputs.push_back(impl);
		}
	}
	Variable ys = forward(inputs);
	
	// TODO: make multiple outputs
	/*
	for (const auto& y : ys) {
		output = std::make_shared<VariableImpl>(y);	
		outputs->creator = shared_from_this();
	}
	*/ 
	auto out = ys.get_impl();
	if (dcz::Config::get().enable_backprop) 
		out->creator = shared_from_this();

	output = out;
	return ys;
}

std::string Function::name() {
	std::string mangled = typeid(*this).name();
	size_t i = 0;
	while (i < mangled.size() && std::isdigit(mangled[i])) {
		i++;
	}
	return mangled.substr(i);
}

Variable Square::forward(const std::vector<Variable>& xs) {
	const Tensor<>& x_data = xs[0].data();
	Tensor<> result = x_data * x_data;
	return Variable(result);
}

std::vector<Variable> Square::backward(const Variable& gy) {
	const Variable& x = inputs[0];
	return { 2.0f * x * gy};
}

Variable Add::forward(const std::vector<Variable>& xs) {
	const Tensor<>& a = xs[0].data();
	const Tensor<>& b = xs[1].data();
	Tensor<> result = a + b;
	return Variable(result);
}

std::vector<Variable> Add::backward(const Variable& gy) {
	return {gy, gy};
}

Variable Mul::forward(const std::vector<Variable>& xs) {
	const Tensor<>& a = xs[0].data();
	const Tensor<>& b = xs[1].data();
	Tensor<> result = a * b;
	return Variable(result);
}

std::vector<Variable> Mul::backward(const Variable& gy) {
	const Variable& a = inputs[0];
	const Variable& b = inputs[1];

	return {b*gy, a*gy};
}

Variable Neg::forward(const std::vector<Variable>& xs) {
	const Tensor<>& x = xs[0].data();
	Tensor<> result = -x;
	return Variable(result);
}

std::vector<Variable> Neg::backward(const Variable& gy) {
	return {-gy};
}

Variable Sub::forward(const std::vector<Variable>& xs) {
	const Tensor<>& a = xs[0].data();
	const Tensor<>& b = xs[1].data();
	Tensor<> result = a - b;
	return Variable(result);
}

std::vector<Variable> Sub::backward(const Variable& gy) {
	return {gy, -gy};
}

Variable Div::forward(const std::vector<Variable>& xs) {
	const Tensor<>& a = xs[0].data();
	const Tensor<>& b = xs[1].data();
	float scalar = b.raw_data()[0];
	Tensor result = a / scalar;
	return Variable(result);
}

std::vector<Variable> Div::backward(const Variable& gy) {
	const Variable& a = inputs[0];
	const Variable& b = inputs[1];
	float scalar = b.data().raw_data()[0];
 
	Variable grad0 = gy / scalar;
	Variable grad1 = -gy * a / (scalar * scalar);
	return {grad0, grad1};
}

Variable Pow::forward(const std::vector<Variable>& xs) {
	const Tensor<>& base = xs[0].data();
	float scalar = xs[1].data().raw_data()[0];

	Tensor result = pow(base, scalar);
	return Variable(result);
}

std::vector<Variable> Pow::backward(const Variable& gy) {
	const Variable& a = inputs[0];
	const Variable& b = inputs[1];
    float scalar = b.data().raw_data()[0];

	Variable result = gy * scalar * (a^(scalar-1));
    return {result};
}

Variable Exp::forward(const std::vector<Variable>& xs) {
    const Tensor<>& x = xs[0].data();
    Tensor result = exp(x);
    return Variable(result);
}

std::vector<Variable> Exp::backward(const Variable& gy) {
	const Variable& x = inputs[0];
	return {gy * exp(x)};
}

Variable Sin::forward(const std::vector<Variable>& xs) {
    const Tensor<>& x = xs[0].data();
    Tensor result = sin(x);
    return Variable(result);
}

std::vector<Variable> Sin::backward(const Variable& gy) {
	const Variable& x = inputs[0];
	return {gy * cos(x)};
}

Variable Cos::forward(const std::vector<Variable>& xs) {
    const Tensor<>& x = xs[0].data();
    Tensor result = cos(x);
    return Variable(result);
}

std::vector<Variable> Cos::backward(const Variable& gy) {
	const Variable& x = inputs[0];
	return {gy * -sin(x)};
}

