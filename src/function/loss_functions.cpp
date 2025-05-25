#include "function/loss_functions.hpp"
#include "container/tensor/tensor_all.hpp"

Variable MeanSquaredError::forward(const std::vector<Variable>& xs) {
	const Tensor<>& a = xs[0].data();
	const Tensor<>& b = xs[1].data();

	const Tensor<> diff = a - b; 
	float scale = static_cast<float>(diff.size());
	const Tensor<> result = (diff^2.0f).sum() / scale;
	return Variable(result);
}

std::vector<Variable> MeanSquaredError::backward(const Variable& gy) {
	Variable x0 = inputs[0];
	Variable x1 = inputs[1];
	Variable diff = x0 - x1;

	float scale = 2.0f / static_cast<float>(diff.size());
	Variable gx0 = gy * diff * scale;
	Variable gx1 = -gx0;
	
	return {gx0, gx1};
}
