#pragma once

#include "container/tensor/tensor.hpp"
#include "ops/ops.hpp"

#include <memory>
#include <vector>
#include <string>
#include <type_traits>

#include <iostream>

class Function; // forward declaration

using namespace tensor;

template<typename T = float>
class VariableImpl {
public:
	//TODO: extend to tensor
    Tensor<T> data;
	std::string name;
    Tensor<T> grad;
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
	Variable operator+(const Variable& rhs) const {
		return add(*this,rhs); }; 
	Variable operator+(const float rhs) const {
		return add(*this,rhs); }; 
	friend Variable operator+(const float lhs, const Variable& rhs) {
		return add(lhs,rhs); };

	Variable operator*(const Variable& rhs) const {
		return mul(*this,rhs); }; 
	Variable operator*(const float rhs) const {
		return mul(*this,rhs); }; 
	friend Variable operator*(const float lhs, const Variable& rhs) {
		return mul(lhs,rhs); };

	friend Variable operator-(const Variable& rhs) {
		return neg(rhs); };

	Variable operator-(const Variable& rhs) const {
		return sub(*this,rhs); };
	Variable operator-(const float rhs) const {
		return sub(*this,rhs); }; 
	friend Variable operator-(const float lhs, const Variable& rhs) {
		return sub(lhs,rhs); };

	Variable operator/(const Variable& rhs) const {
		return div(*this,rhs); };
	Variable operator/(const float rhs) const {
		return div(*this,rhs); }; 
	friend Variable operator/(const float lhs, const Variable& rhs) {
		return div(lhs,rhs); };

	Variable operator^(const float rhs) const {
		return pow(*this,rhs); }; 

public:
// impl functions
	std::string name() const {
		return impl->name; };
	void set_name(std::string s) {
		impl->name = s; };
	const Tensor<>& data() const {return impl->data;};
	const Tensor<>& grad() const {return impl->grad;};
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
	


