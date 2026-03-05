#include "function/rmsnorm_functions.hpp"
#include "container/tensor/tensor_all.hpp"

#include <cmath>

using namespace tensor;

Variable function::RMSNormFunc::forward(const std::vector<Variable>& xs) {
	const Tensor<>& x = xs[0].data();       // [batch, seq_len, hidden_size]
	const Tensor<>& weight = xs[1].data();   // [hidden_size]

	auto shape = x.get_shape();
	size_t batch = shape[0];
	size_t seq_len = shape[1];
	size_t hidden_size = shape[2];

	const auto& x_data = x.raw_data();
	const auto& w_data = weight.raw_data();

	std::vector<float> out_data(batch * seq_len * hidden_size);

	for (size_t b = 0; b < batch; ++b) {
		for (size_t s = 0; s < seq_len; ++s) {
			size_t offset = (b * seq_len + s) * hidden_size;

			// Compute mean of x^2 along last dim
			float sum_sq = 0.0f;
			for (size_t h = 0; h < hidden_size; ++h) {
				float val = x_data[offset + h];
				sum_sq += val * val;
			}
			float mean_sq = sum_sq / static_cast<float>(hidden_size);
			float rsqrt_val = 1.0f / std::sqrt(mean_sq + eps);

			// output = x * rsqrt * weight
			for (size_t h = 0; h < hidden_size; ++h) {
				out_data[offset + h] = x_data[offset + h] * rsqrt_val * w_data[h];
			}
		}
	}

	Tensor<> result(shape, out_data);
	return Variable(result);
}

std::vector<Variable> function::RMSNormFunc::backward(const Variable& gy) {
	// Inference-only: backward not needed
	return {};
}
