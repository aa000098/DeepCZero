#include "function/function.hpp"
#include "config/config.hpp"

#include <cmath>
#include <algorithm>
#include <memory>


Variable Function::operator()(const std::vector<Variable>& inputs) {
	this->inputs.clear();

	std::vector<Tensor<>> xs;
	if (dcz::Config::get().enable_backprop) {
		for (const auto& input : inputs) {
			std::shared_ptr<VariableImpl<>> impl = input.get_impl();
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
	auto out = std::make_shared<VariableImpl<>>(ys);
	if (dcz::Config::get().enable_backprop) out->creator = shared_from_this();
	output = out;
	return Variable(out);
}

Tensor<> Square::forward(std::vector<Tensor<>>& xs) {
	Tensor x = xs[0];
	Tensor result({x.get_shape()}, 0.0f);
	for (size_t i = 0; i < x.size(); ++i)
		result.raw_data()[i] = pow(x.raw_data()[i], 2);	
	return result;
} 

std::vector<Tensor<>> Square::backward(Tensor<>& gy) {
	Tensor x = inputs[0]->data;
	Tensor result({x.get_shape()}, 0.0f);
	for (size_t i = 0; i < x.size(); ++i)
		result.raw_data()[i] = 2 * x.raw_data()[i] * gy.raw_data()[i];
	return {result};
}


Tensor<> Exp::forward(std::vector<Tensor<>>& xs) {
	Tensor x = xs[0];
	Tensor result({x.get_shape()}, 0.0f);
	for (size_t i = 0; i < x.size(); ++i)
		result.raw_data()[i] = exp(x.raw_data()[i]);
	return result;
}

std::vector<Tensor<>> Exp::backward(Tensor<>& gy) {
	Tensor x = inputs[0]->data;
	Tensor result({x.get_shape()}, 0.0f);
	for (size_t i = 0; i < x.size(); ++i)
		result.raw_data()[i] = exp(x.raw_data()[i]) * gy.raw_data()[i];
	return {result};
}

Tensor<> Add::forward(std::vector<Tensor<>>& xs) {
	Tensor a = xs[0];
	Tensor b = xs[1];

	Tensor result({a.get_shape()}, 0.0f);
	for (size_t i = 0; i < a.size(); ++i)
		result.raw_data()[i] = a.raw_data()[i] + b.raw_data()[i];
	return result;
}

std::vector<Tensor<>> Add::backward(Tensor<>& gy) {
	return {gy, gy};
}

Tensor<> Mul::forward(std::vector<Tensor<>>& xs) {
	Tensor a = xs[0];
	Tensor b = xs[1];

	Tensor result({a.get_shape()}, 0.0f);
	for (size_t i = 0; i < a.size(); ++i)
		result.raw_data()[i] = a.raw_data()[i] * b.raw_data()[i];
	return result;
}

std::vector<Tensor<>> Mul::backward(Tensor<>& gy) {
	Tensor a = inputs[0]->data;
	Tensor b = inputs[1]->data;
	
	Tensor result_a({a.get_shape()}, 0.0f);
	Tensor result_b({b.get_shape()}, 0.0f);
	for (size_t i = 0; i < a.size(); ++i) {
		result_a.raw_data()[i] = b.raw_data()[i] * gy.raw_data()[i];
		result_b.raw_data()[i] = a.raw_data()[i] * gy.raw_data()[i];
	}
	return {result_a, result_b};
}

Tensor<> Neg::forward(std::vector<Tensor<>>& xs) {
	Tensor x = xs[0];

	Tensor result({x.get_shape()}, 0.0f);
	for (size_t i = 0; i < x.size(); i++)
		result.raw_data()[i] = x.raw_data()[i] * -1;
	return result;
}

std::vector<Tensor<>> Neg::backward(Tensor<>& gy) {
	Tensor result({gy.get_shape()}, 0.0f);
	for (size_t i = 0; i < gy.size(); i++)
		result.raw_data()[i] = gy.raw_data()[i] * -1;
	return {result};
}

Tensor<> Sub::forward(std::vector<Tensor<>>& xs) {
	Tensor a = xs[0];
	Tensor b = xs[1];

	Tensor result({a.get_shape()}, 0.0f);
	for (size_t i = 0; i < a.size(); ++i)
		result.raw_data()[i] = a.raw_data()[i] - b.raw_data()[i];
	return result;
}


std::vector<Tensor<>> Sub::backward(Tensor<>& gy) {
	Tensor a = inputs[0]->data;
	Tensor b = inputs[1]->data;
	
	Tensor pos_gy = gy;

	std::shared_ptr<Function> f = std::make_shared<Neg>();
	auto gy_vec = std::vector<Tensor<>>();
	gy_vec.push_back(gy);
	Tensor neg_gy = f->forward(gy_vec);

	return {gy, neg_gy};
}
/*
Tensor<> Sub::forward(std::vector<Tensor<>>& xs) {
}
std::vector<Tensor<>> Sub::backward(Tensor<>& gy) {
}
*/

Function::~Function() {}
