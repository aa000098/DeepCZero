#include "function/util_functions.hpp"
#include "container/tensor/tensor_all.hpp"

#include <algorithm>


Variable function::Sum::forward(const std::vector<Variable>& xs) {
	const Tensor<>& x_data = xs[0].data();
	x_shape = x_data.get_shape();

	Tensor<> result = x_data.sum(axis, keepdims);
	return Variable(result);
}

std::vector<Variable> function::Sum::backward(const Variable& gy) {
	
	// reshape_sum_backward
	std::vector<size_t> new_shape;

	if (x_shape.size() == 0 || axis.empty() || keepdims) 
		new_shape = gy.shape();
	else {
		std::vector<int> actual_axis = axis;
		for (auto& a : actual_axis)
			if (a < 0) a += x_shape.size();

	    std::sort(actual_axis.begin(), actual_axis.end()); 

		new_shape = gy.shape();
		for (auto a : actual_axis)
			new_shape.insert(new_shape.begin() + a, 1);
	}

	Variable reshaped = gy.reshape(new_shape);

	// broadcast to
	Variable gx = broadcast_to(reshaped, x_shape);

	return { gx };
}


Variable function::Broadcast_To::forward(const std::vector<Variable>& xs) {
	const Tensor<>& x_data = xs[0].data();
	x_shape = x_data.get_shape();

	Tensor<> result = broadcast_to(x_data, shape);
	return Variable(result);
}

std::vector<Variable> function::Broadcast_To::backward(const Variable& gy) {	
	Variable gx = sum_to(gy, x_shape);

	return { gx };
}


Variable function::Sum_To::forward(const std::vector<Variable>& xs) {
	const Tensor<>& x_data = xs[0].data();
	x_shape = x_data.get_shape();

	Tensor<> result = sum_to(x_data, shape);
	return Variable(result);
}

std::vector<Variable> function::Sum_To::backward(const Variable& gy) {

	Variable reshaped = gy;

	if (gy.shape() != x_shape) {
		std::vector<size_t> new_shape = gy.shape();

		while (new_shape.size() < x_shape.size())
			new_shape.push_back(1);

		reshaped = gy.reshape(new_shape);
	} 

	Variable gx = broadcast_to(reshaped, x_shape);

	return { gx } ;
}



