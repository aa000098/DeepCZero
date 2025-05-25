#include "function/loss_functions.hpp"


Variable MeanSquaredError::forward(const std::vector<Variable>& xs) {
	const 
	Variable diff = xs[0] - xs[1];
	Variable y = sum(diff^2) / diff.size();
	return y;
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
