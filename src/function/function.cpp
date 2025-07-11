#include "function/function.hpp"
#include "config/config.hpp"
#include "container/tensor/tensor_all.hpp"
#include "graph/utils/utils.hpp"

#include <cmath>
#include <algorithm>
#include <memory>

Variable Function::operator()(const std::vector<Variable>& inputs) {
	this->inputs.clear();

	if (dcz::Config::get().enable_backprop) {
		for (const auto& input : inputs) {
			std::shared_ptr<VariableImpl<>> impl = input.get_impl();
			this->inputs.push_back(impl);
		}
	}
	Variable ys = forward(inputs);
	
	// TODO: make multiple outputs
	/*
	for (const auto& y : ys) {
		output = std::make_shared<VariableImpl>(y);	
		outputs->creator = shared_from_this();
	}
	*/ 
	auto out = ys.get_impl();
	if (dcz::Config::get().enable_backprop) 
		out->creator = shared_from_this();

	output = out;
	return ys;
}

Variable Function::operator()(const std::initializer_list<Variable>& inputs) {
		std::vector<Variable> input_vec(inputs);
		return (*this)(input_vec);
}

Variable Function::operator()(const Variable& input) {
	return (*this)({input});
}

std::string Function::name() {
	std::string demangled = demangle(typeid(*this).name());
	return remove_namespace(demangled);
}

void Function::debug_function_refs(std::shared_ptr<Function> f) {
	std::cout << "Function: " << f->name() << std::endl; 

	auto inputs = f->get_inputs();
	for (size_t i = 0; i < inputs.size(); ++i) {
        auto ptr = inputs[i];
        std::cout << "  Input[" << i << "] use_count: " << ptr.use_count() << " id: " << ptr->id() << std::endl;
    }

    auto out = f->get_output();
    if (out) {
        std::cout << "  Output use_count: " << out.use_count() << " id: " << out->id() << std::endl;
    }
}

