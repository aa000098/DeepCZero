#include "function/function.hpp"

#include <cmath>
#include <algorithm>

Variable Function::operator()(const std::vector<Variable>& inputs) {
	this->inputs.clear();

	std::vector<Tensor> xs;
	for (const auto& input : inputs) {
		std::shared_ptr<VariableImpl> impl = input.get_impl();
		this->inputs.push_back(impl);
		xs.push_back(impl->data);
	}
		
	Tensor ys = forward(xs);
	
	// TODO: make multiple outputs
	/*
	for (const auto& y : ys) {
		output = std::make_shared<VariableImpl>(y);	
		outputs->creator = shared_from_this();
	}
	*/ 
	output = std::make_shared<VariableImpl>(ys);
	output->set_creator(shared_from_this());

	return Variable(output);
}

Tensor Square::forward(std::vector<Tensor>& xs) {
	Tensor x = xs[0];
	Tensor result;
	for (auto data : x) {
		float y = pow(data, 2);
		result.push_back(y);
	}
	return result;
}

std::vector<Tensor> Square::backward(Tensor& gy) {
	Tensor x = inputs[0]->data;
	Tensor result;
	for (size_t i = 0; i < x.size(); ++i) {
		float gx = 2 * x[i] * gy[i];
		result.push_back(gx);
	}
	return {result};
}


Tensor Exp::forward(std::vector<Tensor>& xs) {
	Tensor x = xs[0];
	Tensor result;
	for (auto data : x) {
		float y = exp(data);
		result.push_back(y);
	}
	return result;
}

std::vector<Tensor> Exp::backward(Tensor& gy) {
	Tensor x = inputs[0]->data;
	Tensor result;
	for (size_t i = 0; i < x.size(); ++i) {
		float gx = exp(x[i]) * gy[i];
		result.push_back(gx);
	}
	return {result};
}

Tensor Add::forward(std::vector<Tensor>& xs) {
	Tensor a = xs[0];
	Tensor b = xs[1];

	Tensor result;
	for(size_t i = 0; i < a.size(); ++i)
		result.push_back(a[i] + b[i]);
	return result;
}

std::vector<Tensor> Add::backward(Tensor& gy) {
	return {gy, gy};
}

Function::~Function() {}
