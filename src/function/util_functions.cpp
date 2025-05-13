#include "function/util_functions.hpp"

#include <algorithm>

Variable Sum::forward(const std::vector<Variable>& xs) {
	const Tensor<>& x_data = xs[0].data();
	x_shape = x_data.get_shape();

	Tensor<> result = x_data.sum(axis, keepdims);
	return Variable(result);
}

std::vector<Variable> Sum::backward(const Variable& gy) {
	
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

	// broadcast to

	return { reshape(gy, new_shape) };
}


