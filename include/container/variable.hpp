#pragma once

#include "container/tensor/tensorbase.hpp"
#include "container/tensor/tensor1D.hpp"
#include "container/tensor/tensorND.hpp"

#include <memory>
#include <vector>
#include <string>

class Function; // forward declaration

using Tensor = std::shared_ptr<tensor::TensorBase<float>>;

class VariableImpl {
public:
	//TODO: extend to tensor
    Tensor data;
	std::string name;
    Tensor grad;
    std::shared_ptr<Function> creator;
    bool requires_grad;

public:
	VariableImpl(	const Tensor& data, 
					std::string name="", 
					bool requires_grad=true) 
		: data(std::make_shared<tensor::Tensor(data)), name(name), requires_grad(requires_grad) {};

	VariableImpl(	const std::vector<float>& vec, 
					std::string name="", 
					bool requires_grad=true) 
		: data(std::make_shared<tensor::Tensor1D<float>>(vec)), 
		name(name), grad(std::make_shared<tensor::TensorND<float>>(data->shape(), {})), creator(), requires_grad(requires_grad) {};

};


class Variable{
private:
	std::shared_ptr<VariableImpl> impl;

public:
	Variable(	const Tensor& data, 
				std::string name="", 
				bool requires_grad=true) 
		: impl(std::make_shared<VariableImpl>(data, name, requires_grad)) {};
	
	Variable(	const std::vector<float>& vec, 
				std::string name="", 
				bool requires_grad = true) 
		: impl(std::make_shared<VariableImpl>(vec, name, requires_grad)) {};

	Variable(std::shared_ptr<VariableImpl> impl) : impl(std::move(impl)) {};

	const std::shared_ptr<VariableImpl>& get_impl() const { return impl; };

	std::shared_ptr<Function> get_creator() const { return impl->creator; };

    void backward(bool retain_grad=false);

public:

//	float& operator[](size_t idx) {return impl->data[idx]; };
	tensor::TensorView<float> operator[](size_t idx) const { return (*impl->data)[idx]; };
	std::vector<size_t> shape() { return (*impl->data).get_shape(); };
	bool empty() { return (*impl->data).empty(); };
	size_t size() const {return (*impl->data).size(); };
	size_t ndim() const {return (*impl->data).ndim(); };

    void show() const;
};
	


