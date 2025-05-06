#pragma once

#include "container/tensor/tensor.hpp"

#include <memory>
#include <vector>
#include <string>
#include <type_traits>

#include <iostream>

class Function; // forward declaration

using namespace tensor;

class Variable;

template<typename T = float>
class VariableImpl {
public:
	//TODO: extend to tensor
    Tensor<T> data;
	std::string name;
	std::shared_ptr<Variable> grad;
    std::shared_ptr<Function> creator;
    bool requires_grad;

public:
	VariableImpl(	const Tensor<T>& data, 
					std::string name="", 
					bool requires_grad=true) 
		: data(data), 
		name(name), 
		grad(), 
		requires_grad(requires_grad) {};

	VariableImpl(	const std::vector<T>& vec, 
					std::string name="", 
					bool requires_grad=true) 
		: data(Tensor<T>({1}, vec)), 
		name(name), 
		grad(), 
		creator(), 
		requires_grad(requires_grad) {};

	std::uintptr_t id() const {
		return reinterpret_cast<std::uintptr_t>(this);
	};
};


class Variable{
private:
	std::shared_ptr<VariableImpl<>> impl;

public:
	Variable(	const Tensor<>& data, 
				std::string name="", 
				bool requires_grad=true) 
		: impl(std::make_shared<VariableImpl<>>(data, name, requires_grad)) {};
	
	Variable(	const std::vector<float>& vec, 
				std::string name="", 
				bool requires_grad = true) 
		: impl(std::make_shared<VariableImpl<>>(vec, name, requires_grad)) {};

	Variable(	const std::initializer_list<float>& vec,
				std::string name="", 
				bool requires_grad = true) 
		: impl(std::make_shared<VariableImpl<>>(vec, name, requires_grad)) {};

	Variable(std::shared_ptr<VariableImpl<>> impl) 
		: impl(std::move(impl)) {};

	const std::shared_ptr<VariableImpl<>>& get_impl() const { return impl; };

	std::shared_ptr<Function> get_creator() const { 
		return impl->creator; };

    void backward(bool retain_grad=false);

public:
// operator functions
	float& operator()(
			std::vector<size_t> idx) {
		return impl->data(idx); };
	tensor::TensorView<float> operator[](
			size_t idx) const { 
		return impl->data[idx]; };

public:
// impl functions
	std::string name() const {
		return impl->name; };
	void set_name(std::string s) {
		impl->name = s; };
	Tensor<>& data() {return impl->data;};
	const Tensor<>& data() const {return impl->data;};
	const std::shared_ptr<Variable> grad() const {return impl->grad;};
	void cleargrad() {impl->grad.reset();};
	std::string dtype_string() const {return "float";};
	std::uintptr_t id() const {
		return reinterpret_cast<std::uintptr_t>(impl.get());
	}

public:
// override functions
	std::vector<size_t> shape() { 
		return impl->data.get_shape(); };
	std::vector<size_t> shape() const { 
		return impl->data.get_shape(); };
	bool empty() { 
		return impl->data.empty(); };
	size_t size() const {
		return impl->data.size(); };
	size_t ndim() const {
		return impl->data.ndim(); };

    void show() const;
};
	


