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
    : data(data), grad(data.size(), {}), creator(nullptr), requires_grad(requires_grad) {};

	void set_creator(std::shared_ptr<Function> creator) { this->creator = creator; };
};


class Variable{
private:
	std::shared_ptr<VariableImpl> impl;

public:
    Variable(const Tensor& data, bool requires_grad = true);
	Variable(std::shared_ptr<VariableImpl> impl);

	const std::shared_ptr<VariableImpl>& get_impl() const { return impl; };

	std::shared_ptr<Function> get_creator() const { return impl->creator; };
    void set_creator(std::shared_ptr<Function> func) { impl->creator = func; };

    void backward();
    void show() const;
};
	


