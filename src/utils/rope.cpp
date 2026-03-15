#include "utils/rope.hpp"

#include <cmath>
#include <vector>

std::pair<Tensor<>, Tensor<>> precompute_rope_frequencies(
	size_t head_dim, size_t max_seq_len, float theta) {

	size_t half_dim = head_dim / 2;

	// Compute inverse frequencies: freq[i] = 1.0 / (theta^(2i/head_dim))
	std::vector<float> inv_freq(half_dim);
	for (size_t i = 0; i < half_dim; ++i) {
		float exponent = static_cast<float>(2 * i) / static_cast<float>(head_dim);
		inv_freq[i] = 1.0f / std::pow(theta, exponent);
	}

	// Compute cos and sin tables: [max_seq_len, head_dim]
	// Each position t, each pair (2i, 2i+1) uses the same frequency
	std::vector<float> cos_data(max_seq_len * head_dim);
	std::vector<float> sin_data(max_seq_len * head_dim);

	for (size_t t = 0; t < max_seq_len; ++t) {
		for (size_t i = 0; i < half_dim; ++i) {
			float angle = static_cast<float>(t) * inv_freq[i];
			float cos_val = std::cos(angle);
			float sin_val = std::sin(angle);

			// Store for both even and odd positions in the pair
			cos_data[t * head_dim + 2 * i]     = cos_val;
			cos_data[t * head_dim + 2 * i + 1] = cos_val;
			sin_data[t * head_dim + 2 * i]     = sin_val;
			sin_data[t * head_dim + 2 * i + 1] = sin_val;
		}
	}

	Tensor<> cos_table({max_seq_len, head_dim}, cos_data);
	Tensor<> sin_table({max_seq_len, head_dim}, sin_data);

	return {cos_table, sin_table};
}

Variable apply_rope(const Variable& x,
					const Tensor<>& cos_cache,
					const Tensor<>& sin_cache,
					size_t position_offset) {

	// x: [batch, seq_len, num_heads, head_dim]
	auto shape = x.data().get_shape();
	dcz::Device orig_device = x.device();
	size_t batch = shape[0];
	size_t seq_len = shape[1];
	size_t num_heads = shape[2];
	size_t head_dim = shape[3];
	size_t half_dim = head_dim / 2;

	// CPU fallback: raw_data() not available on device tensors
	// Must store CPU copies in locals to avoid dangling references from temporaries
	Tensor<> x_cpu = orig_device.is_cpu() ? x.data() : x.data().cpu();
	Tensor<> cos_cpu = cos_cache.is_cpu() ? cos_cache : cos_cache.cpu();
	Tensor<> sin_cpu = sin_cache.is_cpu() ? sin_cache : sin_cache.cpu();
	const auto& x_data = x_cpu.raw_data();
	const auto& cos_data = cos_cpu.raw_data();
	const auto& sin_data = sin_cpu.raw_data();

	std::vector<float> out_data(batch * seq_len * num_heads * head_dim);

	for (size_t b = 0; b < batch; ++b) {
		for (size_t s = 0; s < seq_len; ++s) {
			size_t pos = position_offset + s;
			size_t cos_offset = pos * head_dim;

			for (size_t h = 0; h < num_heads; ++h) {
				size_t x_offset = ((b * seq_len + s) * num_heads + h) * head_dim;

				// Apply rotation to each pair (x_even, x_odd)
				for (size_t i = 0; i < half_dim; ++i) {
					float x_even = x_data[x_offset + 2 * i];
					float x_odd  = x_data[x_offset + 2 * i + 1];
					float cos_val = cos_data[cos_offset + 2 * i];
					float sin_val = sin_data[cos_offset + 2 * i];

					// Rotation: [cos -sin; sin cos] * [x_even; x_odd]
					out_data[x_offset + 2 * i]     = x_even * cos_val - x_odd * sin_val;
					out_data[x_offset + 2 * i + 1] = x_even * sin_val + x_odd * cos_val;
				}
			}
		}
	}

	Tensor<> result(shape, out_data);
	if (!orig_device.is_cpu()) result = result.to(orig_device);
	return Variable(result);
}
