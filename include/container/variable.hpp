#pragma once

#include <memory>

class Function; // forward declaration

class VariableImpl {
public:
	//TODO: extend to tensor
    float data;
    float grad;
    std::shared_ptr<Function> creator;
    bool requires_grad;

public:
	VariableImpl(float data, bool requires_grad = true);
};


class Variable{
private:
	std::shared_ptr<VariableImpl> impl;

public:
    Variable(float data, bool requires_grad = true);
	Variable(std::shared_ptr<VariableImpl> impl);

	std::shared_ptr<VariableImpl> get_impl() { return impl; };
	float get_data() const { return impl->data; };
	float get_grad() const { return impl->grad; };
	std::shared_ptr<Function> get_creator() const { return impl->creator; };

	void set_grad(float g) {impl->grad = g; }
    void set_creator(std::shared_ptr<Function> func) { impl->creator = func; };
    void backward();
    void show() const;
};
	


