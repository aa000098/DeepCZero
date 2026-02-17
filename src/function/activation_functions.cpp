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


Variable function::SiLU::forward(const std::vector<Variable>& xs) {
	const Tensor<>& x = xs[0].data();
	const Tensor<> sig = tanh(x * 0.5f) * 0.5f + 0.5f;
	const Tensor<> result = x * sig;
	return Variable(result);
}

std::vector<Variable> function::SiLU::backward(const Variable& gy) {
	Variable x(inputs[0]);
	Variable y = output.lock();
	Variable sig_x = sigmoid(x);
	Variable gx = gy * (sig_x + y * (1.0f - sig_x));
	return { gx };
}


Variable function::Dropout::forward(const std::vector<Variable>& xs) {
	const Variable& x = xs[0];

	if (dcz::Config::get().train) {
		Tensor<> mask(x.data().get_shape());
		Tensor<> random = rand(x.data().get_shape());
		for (size_t i = 0; i < x.data().size(); i++)
			mask.raw_data()[i] = random.raw_data()[i] > dropout_rate ? 1.0f : 0.0f;
		float scale = 1 - dropout_rate;
		Tensor<> result = x.data() * mask / scale;
		return Variable(result);
	} else {
		return x;  // 그래프 유지를 위해 입력 Variable 그대로 반환
	}
}


