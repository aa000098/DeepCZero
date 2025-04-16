#pragma once

#include <memory>
#include <vector>

class Function; // forward declaration

using Tensor = std::vector<float>;

class VariableImpl {
public:
	//TODO: extend to tensor
    Tensor data;
    Tensor grad;
    std::shared_ptr<Function> creator;
    bool requires_grad;

public:
	VariableImpl(const Tensor& data, bool requires_grad=true)
    : data(data), grad(data.size(), {}), creator(), requires_grad(requires_grad) {};
};


class Variable{
private:
	std::shared_ptr<VariableImpl> impl;

public:
    Variable(const Tensor& data, bool requires_grad = true);
	Variable(std::shared_ptr<VariableImpl> impl);

	const std::shared_ptr<VariableImpl>& get_impl() const { return impl; };

	std::shared_ptr<Function> get_creator() const { return impl->creator; };

    void backward();
    void show() const;
};
	


