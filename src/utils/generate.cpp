#include "utils/generate.hpp"

#include <iostream>
#include <random>
#include <cmath>
#include <algorithm>

std::vector<int> generate(LlamaForCausalLM& model,
						  const std::vector<int>& prompt_ids,
						  const GenerationConfig& config) {
	model.reset_cache();

	std::vector<int> generated;
	size_t prompt_len = prompt_ids.size();

	std::mt19937 rng(42);

	// 1. Process the full prompt (prefill)
	Variable logits = model.forward_ids(prompt_ids, 0);

	// Get logits for the last position: [1, seq_len, vocab_size] -> last position
	auto logits_shape = logits.shape();
	size_t vocab_size = logits_shape[2];
	size_t seq_len = logits_shape[1];

	// Extract last position logits (CPU fallback for device tensors)
	Tensor<> logits_cpu = logits.data().is_device() ? logits.data().cpu() : logits.data();
	const auto& logits_data = logits_cpu.raw_data();
	size_t last_offset = (seq_len - 1) * vocab_size;

	int next_token;
	if (!config.do_sample || config.temperature <= 0.0f) {
		// Greedy: argmax
		float max_val = logits_data[last_offset];
		next_token = 0;
		for (size_t i = 1; i < vocab_size; ++i) {
			if (logits_data[last_offset + i] > max_val) {
				max_val = logits_data[last_offset + i];
				next_token = static_cast<int>(i);
			}
		}
	} else {
		// Temperature sampling
		std::vector<float> probs(vocab_size);
		float max_logit = *std::max_element(
			logits_data.begin() + last_offset,
			logits_data.begin() + last_offset + vocab_size);

		float sum_exp = 0.0f;
		for (size_t i = 0; i < vocab_size; ++i) {
			probs[i] = std::exp((logits_data[last_offset + i] - max_logit) / config.temperature);
			sum_exp += probs[i];
		}
		for (size_t i = 0; i < vocab_size; ++i) {
			probs[i] /= sum_exp;
		}

		std::discrete_distribution<int> dist(probs.begin(), probs.end());
		next_token = dist(rng);
	}

	generated.push_back(next_token);

	// 2. Autoregressive generation loop
	for (size_t step = 1; step < config.max_new_tokens; ++step) {
		if (next_token == config.eos_token_id) break;

		// Forward with single token, KV cache handles context
		size_t pos_offset = prompt_len + step - 1;
		Variable step_logits = model.forward_ids({next_token}, pos_offset);

		Tensor<> step_cpu = step_logits.data().is_device() ? step_logits.data().cpu() : step_logits.data();
		const auto& step_data = step_cpu.raw_data();
		// logits shape: [1, 1, vocab_size], offset is 0

		if (!config.do_sample || config.temperature <= 0.0f) {
			float max_val = step_data[0];
			next_token = 0;
			for (size_t i = 1; i < vocab_size; ++i) {
				if (step_data[i] > max_val) {
					max_val = step_data[i];
					next_token = static_cast<int>(i);
				}
			}
		} else {
			std::vector<float> probs(vocab_size);
			float max_logit = *std::max_element(step_data.begin(), step_data.begin() + vocab_size);

			float sum_exp = 0.0f;
			for (size_t i = 0; i < vocab_size; ++i) {
				probs[i] = std::exp((step_data[i] - max_logit) / config.temperature);
				sum_exp += probs[i];
			}
			for (size_t i = 0; i < vocab_size; ++i) {
				probs[i] /= sum_exp;
			}

			std::discrete_distribution<int> dist(probs.begin(), probs.end());
			next_token = dist(rng);
		}

		generated.push_back(next_token);
	}

	return generated;
}
