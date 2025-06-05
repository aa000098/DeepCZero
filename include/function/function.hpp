#pragma once

#include "container/variable_all.hpp"

#include <vector>
#include <memory>

namespace function {

class Function : public std::enable_shared_from_this<Function> {
protected:
	std::vector<std::shared_ptr<VariableImpl<>>> inputs;
	std::weak_ptr<VariableImpl<>> output;
	
public:
	virtual Variable operator()(const std::vector<Variable>& inputs);
	virtual Variable operator()(const std::initializer_list<Variable>& inputs);
	virtual Variable operator()(const Variable& input);

	virtual Variable forward(const std::vector<Variable>& xs) = 0;
	virtual std::vector<Variable> backward(const Variable& gy) = 0; 
	virtual std::string name();
	virtual std::uintptr_t id() {
		return reinterpret_cast<std::uintptr_t>(this);
	}

	virtual ~Function() = default;


public:
	std::vector<std::shared_ptr<VariableImpl<>>> get_inputs() { return inputs; }
	std::shared_ptr<VariableImpl<>> get_output() { return output.lock(); };

public:
	void debug_function_refs(std::shared_ptr<Function> f);

};

}
