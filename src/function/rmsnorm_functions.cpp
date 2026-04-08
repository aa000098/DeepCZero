#include "function/rmsnorm_functions.hpp"
#include "container/tensor/tensor_all.hpp"

#include <cmath>

using namespace tensor;

Variable function::RMSNorm::forward(const std::vector<Variable>& xs) {
	const Tensor<>& x = xs[0].data();       // [batch, seq_len, hidden_size]
	const Tensor<>& weight = xs[1].data();   // [hidden_size]

#ifdef USE_SYCL
	if (x.device().type == dcz::DeviceType::SYCL) {
		Tensor<> result = tensor::rmsnorm_forward_sycl(x, weight, eps);
		return Variable(result);
	}
#endif
#ifdef USE_CUDA
	if (x.device().type == dcz::DeviceType::CUDA) {
		Tensor<> result = tensor::rmsnorm_forward_cuda(x, weight, eps);
		return Variable(result);
	}
#endif

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

std::vector<Variable> function::RMSNorm::backward(const Variable& gy) {
	const Tensor<>& x = inputs[0]->data;        // [batch, seq_len, hidden_size]
	const Tensor<>& weight = inputs[1]->data;   // [hidden_size]
	const Tensor<>& gy_data = gy.data();        // [batch, seq_len, hidden_size]

#ifdef USE_SYCL
	if (x.device().type == dcz::DeviceType::SYCL) {
		auto [dx, dw] = tensor::rmsnorm_backward_sycl(x, weight, gy_data, eps);
		return { Variable(dx), Variable(dw) };
	}
#endif
#ifdef USE_CUDA
	if (x.device().type == dcz::DeviceType::CUDA) {
		auto [dx, dw] = tensor::rmsnorm_backward_cuda(x, weight, gy_data, eps);
		return { Variable(dx), Variable(dw) };
	}
#endif

	const auto& shape = x.get_shape();
	size_t batch = shape[0];
	size_t seq_len = shape[1];
	size_t hidden_size = shape[2];

	const auto& x_data = x.raw_data();
	const auto& w_data = weight.raw_data();
	const auto& gy_raw = gy_data.raw_data();

	std::vector<float> dx_data(batch * seq_len * hidden_size, 0.0f);
	std::vector<float> dw_data(hidden_size, 0.0f);

	const float inv_hidden = 1.0f / static_cast<float>(hidden_size);

	for (size_t b = 0; b < batch; ++b) {
		for (size_t s = 0; s < seq_len; ++s) {
			size_t offset = (b * seq_len + s) * hidden_size;

			float sum_sq = 0.0f;
			for (size_t h = 0; h < hidden_size; ++h) {
				float xv = x_data[offset + h];
				sum_sq += xv * xv;
			}
			float mean_sq = sum_sq * inv_hidden;
			float inv_rms = 1.0f / std::sqrt(mean_sq + eps);
			float inv_rms3 = inv_rms * inv_rms * inv_rms;

			float dot = 0.0f;  // sum_h (gy * w * x)
			for (size_t h = 0; h < hidden_size; ++h) {
				dot += gy_raw[offset + h] * w_data[h] * x_data[offset + h];
			}

			for (size_t h = 0; h < hidden_size; ++h) {
				float gxw = gy_raw[offset + h] * w_data[h];
				float xv = x_data[offset + h];
				dx_data[offset + h] = gxw * inv_rms - xv * (inv_rms3 * inv_hidden) * dot;
				dw_data[h] += gy_raw[offset + h] * xv * inv_rms;
			}
		}
	}

	return {
		Variable(Tensor<>(shape, dx_data)),
		Variable(Tensor<>(weight.get_shape(), dw_data))
	};
}
