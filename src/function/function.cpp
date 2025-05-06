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
	const Tensor<>& x_data = inputs[0]->data;
	const Tensor<>& gy_data = gy.data();
	Tensor<> result = 2.0f * x_data * gy_data;
	return {Variable(result)};
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
	const Tensor<>& a = inputs[0]->data;
	const Tensor<>& b = inputs[1]->data;
	const Tensor<>& gy_data = gy.data();

	Tensor<> grad0 = b * gy_data;
	Tensor<> grad1 = a * gy_data;
	return {Variable(grad0), Variable(grad1)};
}

Variable Neg::forward(const std::vector<Variable>& xs) {
	const Tensor<>& x = xs[0].data();
	Tensor<> result = -x;
	return Variable(result);
}

std::vector<Variable> Neg::backward(const Variable& gy) {
	const Tensor<>& gy_data = gy.data();
	Tensor<> result = -gy_data;
	return {Variable(result)};
}

Variable Sub::forward(const std::vector<Variable>& xs) {
	const Tensor<>& a = xs[0].data();
	const Tensor<>& b = xs[1].data();
	Tensor<> result = a - b;
	return Variable(result);
}

std::vector<Variable> Sub::backward(const Variable& gy) {
	const Tensor<>& gy_data = gy.data();
	Tensor<> grad0 = gy_data;
	Tensor<> grad1 = -gy_data;
	return {Variable(grad0), Variable(grad1)};
}

Variable Div::forward(const std::vector<Variable>& xs) {
	const Tensor<>& a = xs[0].data();
	const Tensor<>& b = xs[1].data();
	float scalar = b.raw_data()[0];
	Tensor<> result = a / scalar;
	return Variable(result);
}

std::vector<Variable> Div::backward(const Variable& gy) {
	const Tensor<>& a = inputs[0]->data;
	const Tensor<>& b = inputs[1]->data;
	float scalar = b.raw_data()[0];
	const Tensor<>& gy_data = gy.data();
 
	Tensor<> grad0 = gy_data / scalar;
	Tensor<> grad1 = -gy_data * a / (scalar * scalar);
	return {Variable(grad0), Variable(grad1)};
}

Variable Pow::forward(const std::vector<Variable>& xs) {
	const Tensor<>& base = xs[0].data();
	float scalar = xs[1].data().raw_data()[0];

    Tensor<> result(base.get_shape(), 0.0f);
    for (size_t i = 0; i < base.size(); ++i) 
		result.raw_data()[i] = std::pow(base.raw_data()[i], scalar);
	return Variable(result);
}

std::vector<Variable> Pow::backward(const Variable& gy) {
    const Tensor<>& x_data = inputs[0]->data;
    float scalar = inputs[1]->data.raw_data()[0];
    const Tensor<>& gy_data = gy.data();

    Tensor<> result(x_data.get_shape(), 0.0f);
    for (size_t i = 0; i < x_data.size(); ++i) 
        result.raw_data()[i] = gy_data.raw_data()[i] * scalar * std::pow(x_data.raw_data()[i], scalar-1);
    return {Variable(result)};
}

Variable Exp::forward(const std::vector<Variable>& xs) {
    Variable x = xs[0];
	const Tensor<>& data = x.data();
    Tensor<> result(data.get_shape(), 0.0f);
    for (size_t i = 0; i < data.size(); ++i) 
        result.raw_data()[i] = std::exp(data.raw_data()[i]);
    return Variable(result);
}

std::vector<Variable> Exp::backward(const Variable& gy) {
    Variable x(inputs[0]);
    const Tensor<>& x_data = x.data();
    const Tensor<>& gy_data = gy.data();
    Tensor<> result(x_data.get_shape(), 0.0f);
    for (size_t i = 0; i < x_data.size(); ++i) 
        result.raw_data()[i] = gy_data.raw_data()[i] * std::exp(x_data.raw_data()[i]);
    return {Variable(result)};
}

Variable Sin::forward(const std::vector<Variable>& xs) {
    Variable x = xs[0];
    const Tensor<>& data = x.data();
    Tensor<> result(data.get_shape(), 0.0f);
    for (size_t i = 0; i < data.size(); ++i) 
        result.raw_data()[i] = std::sin(data.raw_data()[i]);
    return Variable(result);
}

std::vector<Variable> Sin::backward(const Variable& gy) {
    const Tensor<>& gy_data = gy.data();
    Variable x(inputs[0]);
    const Tensor<>& x_data = x.data();
    Tensor<> result(gy_data.get_shape(), 0.0f);
    for (size_t i = 0; i < gy_data.size(); ++i) 
        result.raw_data()[i] = gy_data.raw_data()[i] * std::cos(x_data.raw_data()[i]);
    return {Variable(result)};
}

Variable Cos::forward(const std::vector<Variable>& xs) {
    Variable x = xs[0];
    const Tensor<>& data = x.data();
    Tensor<> result(data.get_shape(), 0.0f);
    for (size_t i = 0; i < data.size(); ++i) 
        result.raw_data()[i] = std::cos(data.raw_data()[i]);
    return Variable(result);
}

std::vector<Variable> Cos::backward(const Variable& gy) {
    const Tensor<>& gy_data = gy.data();
    Variable x(inputs[0]);
    const Tensor<>& x_data = x.data();
    Tensor<> result(gy_data.get_shape(), 0.0f);
    for (size_t i = 0; i < gy_data.size(); ++i) 
        result.raw_data()[i] = -gy_data.raw_data()[i] * std::sin(x_data.raw_data()[i]);
    return {Variable(result)};
}

