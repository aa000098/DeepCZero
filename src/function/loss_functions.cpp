#include "function/loss_functions.hpp"
#include "container/tensor/tensor_all.hpp"


Variable function::MeanSquaredError::forward(const std::vector<Variable>& xs) {
	const Tensor<>& a = xs[0].data();
	const Tensor<>& b = xs[1].data();

	const Tensor<> diff = a - b; 
	float scale = static_cast<float>(diff.size());
	const Tensor<> result = (diff^2.0f).sum() / scale;
	return Variable(result);
}

std::vector<Variable> function::MeanSquaredError::backward(const Variable& gy) {
	const Variable& x0 = inputs[0];
	const Variable& x1 = inputs[1];
	Variable diff = x0 - x1;

	float scale = 2.0f / static_cast<float>(diff.size());
	Variable gx0 = gy * diff * scale;
	Variable gx1 = -gx0;
	
	return {gx0, gx1};
}


Variable function::SoftmaxCrossEntropyError::forward(const std::vector<Variable>& xs) {
	const Tensor<>& x = xs[0].data();
	const Tensor<>& t = xs[1].data();
	float N = x.get_shape()[0];	

	// logsumexp
	Tensor<> m = x.max({1}, true);
	Tensor<> y = exp(x - m);
	Tensor<> sum_y = y.sum({1}, true);
	sum_y = log(sum_y);

	Tensor<> log_z = m + sum_y;
	Tensor<> log_p = x - log_z;

	float loss = 0.0f;
	for (size_t i = 0; i < N; i++) {
		size_t label = static_cast<size_t>(t({i}));
		loss += -log_p({i, label});
	}

	loss /= N;

	return Variable(Tensor<>(1, loss));	
}

std::vector<Variable> function::SoftmaxCrossEntropyError::backward(const Variable& gy) {
	const Variable& x = inputs[0];
	const Variable& t = inputs[1];
	size_t N = x.shape()[0];
	size_t C = x.shape()[1];

	Variable y = softmax(x);

	Tensor<> t_onehot({N, C}, 0.0f);

	for (size_t i = 0; i < N; i++) {
		size_t label = t.data()({i});
		t_onehot({i, label}) = 1.0f;
	}

	Variable dx = (y - Variable(t_onehot)) * (gy / static_cast<float>(N));

	return { dx };
}

