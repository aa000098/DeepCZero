#pragma once

#include "container/layer/llama.hpp"
#include <vector>
#include <cstddef>

struct GenerationConfig {
	size_t max_new_tokens = 128;
	float temperature = 1.0f;
	bool do_sample = false;  // false = greedy (argmax), true = sampling
	int eos_token_id = 128009;  // <|eot_id|> for Llama 3.2
};

// Generate token IDs autoregressively
// Returns only the generated tokens (not including the prompt)
std::vector<int> generate(LlamaForCausalLM& model,
						  const std::vector<int>& prompt_ids,
						  const GenerationConfig& config = {});
