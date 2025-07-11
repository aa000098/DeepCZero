#include "function/activation_functions.hpp"
#include "container/tensor/tensor_all.hpp"
#include "config/config.hpp"

Variable function::Sigmoid::forward(const std::vector<Variable>& xs) {
	const Tensor<>& x = xs[0].data();
	const Tensor<> result = tanh(x * 0.5f) * 0.5f + 0.5f;
	return Variable(result);
}

std::vector<Variable> function::Sigmoid::backward(const Variable& gy) {
	Variable y = output.lock();

	Variable gx = gy * y * (1.0f - y);
	return { gx };
}

Variable function::Softmax::forward(const std::vector<Variable>& xs) {
	const Tensor<>& x = xs[0].data();

	Tensor y = x - x.max(axes, true);
	y = exp(y);
	Tensor<> sum_y = y.sum(axes, true);
	return y / sum_y;
}

std::vector<Variable> function::Softmax::backward(const Variable& gy) {
	Variable y = output.lock();

	Variable gx = y * gy;
	Variable sumdx = sum(gx, axes, true);
	gx -= y * sumdx;
	return { gx };

}


Variable function::ReLU::forward(const std::vector<Variable>& xs) {
	const Tensor<>& x = xs[0].data();
	const Tensor<> result = maximum(x, 0.0f);	
	return Variable(result);
}


std::vector<Variable> function::ReLU::backward(const Variable& gy) {
	const Variable& x = inputs[0];
	const Tensor<>& x_data = x.data();

	Tensor<> mask_data = greater(x_data, 0.0f);
	Variable mask(mask_data, "mask");

	Variable gx = gy * mask;
	return { gx };
}


Variable function::Dropout::forward(const std::vector<Variable>& xs) {
	const Tensor<>& x = xs[0].data();

	if (dcz::Config::get().train) {
		Tensor<> mask(x.get_shape());
		Tensor<> random = rand(x.get_shape());
		for (size_t i = 0; i < x.size(); i++)
			mask.raw_data()[i] = random.raw_data()[i] > dropout_rate ? 1.0f : 0.0f;
		float scale = 1 - dropout_rate;
		Tensor<> result = x * mask / scale;
		return Variable(result);
	} else {
		return Variable(x);
	}
}


