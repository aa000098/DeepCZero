#include "function/function.hpp"
#include "config/config.hpp"

#include <cmath>
#include <algorithm>


Variable Function::operator()(const std::vector<Variable>& inputs) {
	this->inputs.clear();

	std::vector<Tensor> xs;
	if (dcz::Config::get().enable_backprop) {
		for (const auto& input : inputs) {
			std::shared_ptr<VariableImpl> impl = input.get_impl();
			this->inputs.push_back(impl);
			xs.push_back(impl->data);
		}
	} else {
		for (const auto& input : inputs) {
			xs.push_back(input.get_impl()->data);
		}
	}
	Tensor ys = forward(xs);
	
	// TODO: make multiple outputs
	/*
	for (const auto& y : ys) {
		output = std::make_shared<VariableImpl>(y);	
		outputs->creator = shared_from_this();
	}
	*/ 
	auto out = std::make_shared<VariableImpl>(ys);
	if (dcz::Config::get().enable_backprop) out->creator = shared_from_this();
	output = out;
	return Variable(out);
}

Tensor Square::forward(std::vector<Tensor>& xs) {
	Tensor x = xs[0];
	Tensor result({x.size()}, 0.0f);
	for (size_t i = 0; i < x.size(); ++i)
		result[i] = pow(x[i], 2);	
	return result;
}

std::vector<Tensor> Square::backward(Tensor& gy) {
	Tensor x = inputs[0]->data;
	Tensor result({x.size()}, 0.0f);
	for (size_t i = 0; i < x.size(); ++i)
		result[i] = 2 * x[i] * gy[i];
	return {result};
}


Tensor Exp::forward(std::vector<Tensor>& xs) {
	Tensor x = xs[0];
	Tensor result({x.size()}, 0.0f);
	for (size_t i = 0; i < x.size(); ++i)
		result[i] = exp(x[i]);
	return result;
}

std::vector<Tensor> Exp::backward(Tensor& gy) {
	Tensor x = inputs[0]->data;
	Tensor result({x.size()}, 0.0f);
	for (size_t i = 0; i < x.size(); ++i)
		result[i] = exp(x[i]) * gy[i];
	return {result};
}

Tensor Add::forward(std::vector<Tensor>& xs) {
	Tensor a = xs[0];
	Tensor b = xs[1];

	Tensor result({a.size()}, 0.0f);
	for (size_t i = 0; i < a.size(); ++i)
		result[i] = a[i] + b[i];
	return result;
}

std::vector<Tensor> Add::backward(Tensor& gy) {
	return {gy, gy};
}

Function::~Function() {}
