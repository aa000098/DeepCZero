#pragma once

#include "container/tensor/tensor_all.hpp"
#include "container/variable.hpp"

#include <utility>
#include <cstddef>

// Precompute cos and sin tables for Rotary Position Embedding
// Returns (cos_table, sin_table), each of shape [max_seq_len, head_dim]
std::pair<Tensor<>, Tensor<>> precompute_rope_frequencies(
	size_t head_dim,
	size_t max_seq_len,
	float theta = 500000.0f);

// Apply RoPE to a tensor
// x: [batch, seq_len, num_heads, head_dim]
// cos_cache, sin_cache: [max_seq_len, head_dim] (from precompute_rope_frequencies)
// position_offset: starting position index (for KV cache / autoregressive generation)
// Returns: rotated tensor of same shape
Variable apply_rope(const Variable& x,
					const Tensor<>& cos_cache,
					const Tensor<>& sin_cache,
					size_t position_offset = 0);
