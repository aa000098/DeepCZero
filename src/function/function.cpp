#include "function/function.hpp"
#include "config/config.hpp"

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
    Variable x = xs[0];
    return x * x;
} 

std::vector<Variable> Square::backward(const Variable& gy) {
    Variable x(inputs[0]);
    Variable gx = 2.0f * x * gy;
    return {gx};
}

Variable Add::forward(const std::vector<Variable>& xs) {
    return xs[0] + xs[1];
}

std::vector<Variable> Add::backward(const Variable& gy) {
    return {gy, gy};
}

Variable Mul::forward(const std::vector<Variable>& xs) {
    return xs[0] * xs[1];
}

std::vector<Variable> Mul::backward(const Variable& gy) {
    Variable a(inputs[0]);
    Variable b(inputs[1]);
    return {b * gy, a * gy};
}

Variable Neg::forward(const std::vector<Variable>& xs) {
    return -xs[0];
}

std::vector<Variable> Neg::backward(const Variable& gy) {
    return {-gy};
}

Variable Sub::forward(const std::vector<Variable>& xs) {
    return xs[0] - xs[1];
}

std::vector<Variable> Sub::backward(const Variable& gy) {
    return {gy, -gy};
}

Variable Div::forward(const std::vector<Variable>& xs) {
    return xs[0] / xs[1];
}

std::vector<Variable> Div::backward(const Variable& gy) {
    Variable a(inputs[0]);
    Variable b(inputs[1]);
	float scalar = b.data().raw_data()[0];
    Variable result_0 = gy / scalar;
    Variable result_1 = -gy * a / (scalar * scalar);
    return {result_0, result_1};
}

Variable Pow::forward(const std::vector<Variable>& xs) {
	float scalar = xs[1].data().raw_data()[0];
    return xs[0]^scalar;
}

std::vector<Variable> Pow::backward(const Variable& gy) {
    Variable a(inputs[0]);
    Variable b(inputs[1]);
	float scalar = b.data().raw_data()[0];
    Variable gx = scalar * (a^(scalar - 1)) * gy;
    return {gx};
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

