#include "container/layer/llama.hpp"
#include "function/rmsnorm_functions.hpp"
#include "function/ops/ops_all.hpp"
#include "container/variable_ops.hpp"
#include "utils/rope.hpp"
#include "cnpy.h"

#include "config/config.hpp"

#include <cmath>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <iomanip>

using namespace tensor;

namespace layer {

// ============================================================
// Embedding
// ============================================================

Embedding::Embedding(size_t vocab_size, size_t embed_dim)
	: embed_dim(embed_dim) {
	// Weight: [vocab_size, embed_dim], initialized to small random values
	Tensor<> w_data({vocab_size, embed_dim}, 0.0f);
	// Simple initialization: scale by 1/sqrt(embed_dim)
	float scale = 1.0f / std::sqrt(static_cast<float>(embed_dim));
	auto& raw = w_data.raw_data();
	for (size_t i = 0; i < raw.size(); ++i) {
		// Deterministic pseudo-random init for reproducibility
		raw[i] = scale * (static_cast<float>((i * 2654435761u) % 1000) / 500.0f - 1.0f);
	}
	Parameter W(w_data, "W");
	register_params("W", W);
}

Variable Embedding::forward_ids(const std::vector<int>& token_ids) {
	const Tensor<>& W = get_param("W").data();

	// Convert int -> size_t for gather_rows
	std::vector<size_t> indices(token_ids.begin(), token_ids.end());

	// gather_rows: select rows from W -> [seq_len, embed_dim]
	Tensor<> selected = W.gather_rows(indices).contiguous();

	// Reshape to [1, seq_len, embed_dim]
	size_t seq_len = token_ids.size();
	Tensor<> result = selected.reshape({1, seq_len, embed_dim});

	return Variable(result);
}

Variable Embedding::forward(const std::vector<Variable>& xs) {
	(void)xs;
	throw std::runtime_error("Embedding::forward not supported. Use forward_ids().");
}

void Embedding::load_from_npz(const cnpy::npz_t& npz, const std::string& prefix) {
	std::string w_key = prefix + ".W";
	auto it = npz.find(w_key);
	if (it == npz.end()) {
		throw std::runtime_error("Embedding weight not found: " + w_key);
	}
	const cnpy::NpyArray& arr = it->second;
	std::vector<float> data = arr.as_vec<float>();
	Tensor<> w_tensor(arr.shape, data);
	set_param_data("W", w_tensor);
}

Tensor<> Embedding::get_weight() const {
	return get_param("W").data();
}

// ============================================================
// LlamaRMSNorm
// ============================================================

LlamaRMSNorm::LlamaRMSNorm(size_t hidden_size, float eps)
	: eps(eps) {
	// Weight initialized to 1.0
	Tensor<> w_data(hidden_size, 1.0f);
	Parameter weight(w_data, "weight");
	register_params("weight", weight);
}

Variable LlamaRMSNorm::forward(const std::vector<Variable>& xs) {
	const Variable& x = xs[0];
	const Parameter& weight = get_param("weight");

	auto f = std::make_shared<function::RMSNorm>(eps);
	return (*f)({x, weight});
}

void LlamaRMSNorm::load_from_npz(const cnpy::npz_t& npz, const std::string& prefix) {
	std::string key = prefix + ".weight";
	auto it = npz.find(key);
	if (it == npz.end()) {
		throw std::runtime_error("RMSNorm weight not found: " + key);
	}
	const cnpy::NpyArray& arr = it->second;
	std::vector<float> data = arr.as_vec<float>();
	Tensor<> w_tensor(arr.shape, data);
	set_param_data("weight", w_tensor);
}

// ============================================================
// LlamaAttention
// ============================================================

LlamaAttention::LlamaAttention(size_t hidden_size, size_t num_heads, size_t num_kv_heads)
	: hidden_size(hidden_size), num_heads(num_heads), num_kv_heads(num_kv_heads),
	  head_dim(hidden_size / num_heads), num_kv_groups(num_heads / num_kv_heads) {

	q_proj = std::make_shared<Linear>(num_heads * head_dim, /*nobias=*/true, hidden_size);
	k_proj = std::make_shared<Linear>(num_kv_heads * head_dim, /*nobias=*/true, hidden_size);
	v_proj = std::make_shared<Linear>(num_kv_heads * head_dim, /*nobias=*/true, hidden_size);
	o_proj = std::make_shared<Linear>(hidden_size, /*nobias=*/true, num_heads * head_dim);

	register_sublayers("q_proj", q_proj);
	register_sublayers("k_proj", k_proj);
	register_sublayers("v_proj", v_proj);
	register_sublayers("o_proj", o_proj);
}

Variable LlamaAttention::forward_attn(const Variable& hidden_states,
									   const Tensor<>& cos_cache,
									   const Tensor<>& sin_cache,
									   size_t position_offset) {
	auto shape = hidden_states.shape();
	size_t batch = shape[0];
	size_t seq_len = shape[1];

	// 1. Project Q, K, V
	Variable Q = (*q_proj)(hidden_states);  // [batch, seq, num_heads * head_dim]
	Variable K = (*k_proj)(hidden_states);  // [batch, seq, num_kv_heads * head_dim]
	Variable V = (*v_proj)(hidden_states);  // [batch, seq, num_kv_heads * head_dim]

	// 2. Reshape to multi-head
	Q = reshape(Q, {batch, seq_len, num_heads, head_dim});
	K = reshape(K, {batch, seq_len, num_kv_heads, head_dim});
	V = reshape(V, {batch, seq_len, num_kv_heads, head_dim});

	// 3. Apply RoPE to Q and K
	Q = apply_rope(Q, cos_cache, sin_cache, position_offset);
	K = apply_rope(K, cos_cache, sin_cache, position_offset);

	// 4. KV Cache: append new K, V to pre-allocated cache (always CPU)
	dcz::Device orig_device = hidden_states.device();
	Tensor<> K_data = K.data().contiguous();
	Tensor<> V_data = V.data().contiguous();
	// KV cache stays on CPU; move K/V to CPU for cache writes
	if (!orig_device.is_cpu()) {
		K_data = K_data.cpu();
		V_data = V_data.cpu();
	}

	size_t kv_stride = num_kv_heads * head_dim;

	if (cache_max_len == 0) {
		// First call: pre-allocate for max sequence length (512)
		cache_max_len = 512;
		k_cache = Tensor<>({batch, cache_max_len, num_kv_heads, head_dim}, 0.0f);
		v_cache = Tensor<>({batch, cache_max_len, num_kv_heads, head_dim}, 0.0f);
	}

	// Write new K,V at cache_len offset (no reallocation needed)
	auto& k_buf = k_cache.raw_data();
	auto& v_buf = v_cache.raw_data();
	const auto& cur_k = K_data.raw_data();
	const auto& cur_v = V_data.raw_data();

	for (size_t b = 0; b < batch; ++b) {
		std::copy(cur_k.begin() + b * seq_len * kv_stride,
				  cur_k.begin() + (b + 1) * seq_len * kv_stride,
				  k_buf.begin() + (b * cache_max_len + cache_len) * kv_stride);
		std::copy(cur_v.begin() + b * seq_len * kv_stride,
				  cur_v.begin() + (b + 1) * seq_len * kv_stride,
				  v_buf.begin() + (b * cache_max_len + cache_len) * kv_stride);
	}

	size_t total_len = cache_len + seq_len;
	cache_len = total_len;

	// 5. Extract valid cache portion [batch, total_len, num_kv_heads, head_dim]
	Tensor<> k_valid, v_valid;
	if (total_len == cache_max_len) {
		k_valid = k_cache;
		v_valid = v_cache;
	} else {
		// Slice valid portion from pre-allocated cache
		std::vector<float> k_slice(batch * total_len * kv_stride);
		std::vector<float> v_slice(batch * total_len * kv_stride);
		for (size_t b = 0; b < batch; ++b) {
			std::copy(k_buf.begin() + b * cache_max_len * kv_stride,
					  k_buf.begin() + (b * cache_max_len + total_len) * kv_stride,
					  k_slice.begin() + b * total_len * kv_stride);
			std::copy(v_buf.begin() + b * cache_max_len * kv_stride,
					  v_buf.begin() + (b * cache_max_len + total_len) * kv_stride,
					  v_slice.begin() + b * total_len * kv_stride);
		}
		k_valid = Tensor<>({batch, total_len, num_kv_heads, head_dim}, k_slice);
		v_valid = Tensor<>({batch, total_len, num_kv_heads, head_dim}, v_slice);
	}

	// Move KV cache results back to device for attention computation
	if (!orig_device.is_cpu()) {
		k_valid = k_valid.to(orig_device);
		v_valid = v_valid.to(orig_device);
	}

	// 5b. GQA: expand KV heads to match query heads
	// [batch, total_len, num_kv_heads, head_dim] -> [batch, total_len, num_heads, head_dim]
	Tensor<> K_full, V_full;
	if (num_kv_groups == 1) {
		K_full = k_valid;
		V_full = v_valid;
	} else {
		Tensor<> k_expanded = k_valid.reshape({batch, total_len, num_kv_heads, 1, head_dim});
		k_expanded = broadcast_to(k_expanded, {batch, total_len, num_kv_heads, num_kv_groups, head_dim}).contiguous();
		K_full = k_expanded.reshape({batch, total_len, num_heads, head_dim});

		Tensor<> v_expanded = v_valid.reshape({batch, total_len, num_kv_heads, 1, head_dim});
		v_expanded = broadcast_to(v_expanded, {batch, total_len, num_kv_heads, num_kv_groups, head_dim}).contiguous();
		V_full = v_expanded.reshape({batch, total_len, num_heads, head_dim});
	}

	// 6. Transpose for attention: [batch, heads, seq/total, head_dim]
	Variable Q_var(Q.data().transpose({0, 2, 1, 3}).contiguous());      // [B, heads, seq, head_dim]
	Variable K_var(K_full.transpose({0, 2, 1, 3}).contiguous());         // [B, heads, total, head_dim]
	Variable V_var(V_full.transpose({0, 2, 1, 3}).contiguous());         // [B, heads, total, head_dim]

	// 7. Scaled dot-product attention
	// K^T: transpose last two dims [B, heads, head_dim, total]
	Variable K_T(K_var.data().transpose({0, 1, 3, 2}).contiguous());

	// scores = Q @ K^T / sqrt(head_dim)
	Variable scores = matmul(Q_var, K_T);
	float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
	scores = scores * scale;

	// Apply causal mask (only needed when seq_len > 1, i.e., prompt processing)
	if (seq_len > 1) {
		// Create causal mask: [seq_len, total_len]
		// For prompt processing (no prior cache), total_len == seq_len
		std::vector<float> mask_data(seq_len * total_len, 0.0f);
		for (size_t i = 0; i < seq_len; ++i) {
			size_t row_offset = position_offset + i;
			for (size_t j = 0; j < total_len; ++j) {
				if (j > row_offset) {
					mask_data[i * total_len + j] = -1e9f;
				}
			}
		}
		Tensor<> mask_2d({seq_len, total_len}, mask_data);
		// Broadcast to [batch, heads, seq_len, total_len]
		Tensor<> mask_4d = mask_2d.reshape({1, 1, seq_len, total_len});
		mask_4d = broadcast_to(mask_4d, {batch, num_heads, seq_len, total_len}).contiguous();
		if (!orig_device.is_cpu()) mask_4d = mask_4d.to(orig_device);
		Variable mask_var(mask_4d);
		scores = scores + mask_var;
	}

	// softmax along last axis
	Variable attn_weights = softmax(scores, {-1});

	// attn_output = attn_weights @ V
	Variable attn_output = matmul(attn_weights, V_var);  // [B, heads, seq, head_dim]

	// 8. Transpose back and reshape
	// [B, heads, seq, head_dim] -> [B, seq, heads, head_dim] -> [B, seq, hidden_size]
	Tensor<> out_t = attn_output.data().transpose({0, 2, 1, 3}).contiguous();
	Variable out_reshaped(out_t.reshape({batch, seq_len, hidden_size}));

	// 9. Output projection
	return (*o_proj)(out_reshaped);
}

Variable LlamaAttention::forward(const std::vector<Variable>& xs) {
	(void)xs;
	throw std::runtime_error("LlamaAttention::forward not supported. Use forward_attn().");
}

void LlamaAttention::reset_cache() {
	k_cache = Tensor<>();
	v_cache = Tensor<>();
	cache_len = 0;
	cache_max_len = 0;
}

void LlamaAttention::load_from_npz(const cnpy::npz_t& npz, const std::string& prefix) {
	q_proj->load_params_from_npz(npz, prefix + ".q_proj");
	k_proj->load_params_from_npz(npz, prefix + ".k_proj");
	v_proj->load_params_from_npz(npz, prefix + ".v_proj");
	o_proj->load_params_from_npz(npz, prefix + ".o_proj");
}

// ============================================================
// LlamaMLP
// ============================================================

LlamaMLP::LlamaMLP(size_t hidden_size, size_t intermediate_size) {
	gate_proj = std::make_shared<Linear>(intermediate_size, /*nobias=*/true, hidden_size);
	up_proj = std::make_shared<Linear>(intermediate_size, /*nobias=*/true, hidden_size);
	down_proj = std::make_shared<Linear>(hidden_size, /*nobias=*/true, intermediate_size);

	register_sublayers("gate_proj", gate_proj);
	register_sublayers("up_proj", up_proj);
	register_sublayers("down_proj", down_proj);
}

Variable LlamaMLP::forward(const std::vector<Variable>& xs) {
	const Variable& x = xs[0];
	// SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))
	Variable gate = silu((*gate_proj)(x));
	Variable up = (*up_proj)(x);
	Variable hidden = gate * up;
	return (*down_proj)(hidden);
}

void LlamaMLP::load_from_npz(const cnpy::npz_t& npz, const std::string& prefix) {
	gate_proj->load_params_from_npz(npz, prefix + ".gate_proj");
	up_proj->load_params_from_npz(npz, prefix + ".up_proj");
	down_proj->load_params_from_npz(npz, prefix + ".down_proj");
}

// ============================================================
// LlamaDecoderLayer
// ============================================================

LlamaDecoderLayer::LlamaDecoderLayer(size_t hidden_size, size_t num_heads,
									   size_t num_kv_heads, size_t intermediate_size,
									   float rms_norm_eps) {
	self_attn = std::make_shared<LlamaAttention>(hidden_size, num_heads, num_kv_heads);
	mlp = std::make_shared<LlamaMLP>(hidden_size, intermediate_size);
	input_layernorm = std::make_shared<LlamaRMSNorm>(hidden_size, rms_norm_eps);
	post_attention_layernorm = std::make_shared<LlamaRMSNorm>(hidden_size, rms_norm_eps);

	register_sublayers("self_attn", self_attn);
	register_sublayers("mlp", mlp);
	register_sublayers("input_layernorm", input_layernorm);
	register_sublayers("post_attention_layernorm", post_attention_layernorm);
}

Variable LlamaDecoderLayer::forward_with_cache(const Variable& hidden_states,
												const Tensor<>& cos_cache,
												const Tensor<>& sin_cache,
												size_t position_offset) {
	// Pre-norm architecture
	Variable residual = hidden_states;
	Variable normed = (*input_layernorm)(hidden_states);
	Variable attn_out = self_attn->forward_attn(normed, cos_cache, sin_cache, position_offset);
	Variable h = residual + attn_out;

	residual = h;
	normed = (*post_attention_layernorm)(h);
	Variable mlp_out = (*mlp)(normed);
	h = residual + mlp_out;

	return h;
}

Variable LlamaDecoderLayer::forward(const std::vector<Variable>& xs) {
	(void)xs;
	throw std::runtime_error("LlamaDecoderLayer::forward not supported. Use forward_with_cache().");
}

void LlamaDecoderLayer::reset_cache() {
	self_attn->reset_cache();
}

void LlamaDecoderLayer::load_from_npz(const cnpy::npz_t& npz, const std::string& prefix) {
	self_attn->load_from_npz(npz, prefix + ".self_attn");
	mlp->load_from_npz(npz, prefix + ".mlp");
	input_layernorm->load_from_npz(npz, prefix + ".input_layernorm");
	post_attention_layernorm->load_from_npz(npz, prefix + ".post_attention_layernorm");
}

// ============================================================
// LlamaModel
// ============================================================

LlamaModel::LlamaModel(size_t vocab_size, size_t hidden_size, size_t num_layers,
						 size_t num_heads, size_t num_kv_heads, size_t intermediate_size,
						 size_t max_position_embeddings,
						 float rope_theta, float rms_norm_eps)
	: num_layers(num_layers) {

	embed_tokens = std::make_shared<Embedding>(vocab_size, hidden_size);
	register_sublayers("embed_tokens", embed_tokens);

	for (size_t i = 0; i < num_layers; ++i) {
		auto layer = std::make_shared<LlamaDecoderLayer>(
			hidden_size, num_heads, num_kv_heads, intermediate_size, rms_norm_eps);
		layers.push_back(layer);
		register_sublayers("layers." + std::to_string(i), layer);
	}

	norm = std::make_shared<LlamaRMSNorm>(hidden_size, rms_norm_eps);
	register_sublayers("norm", norm);

	// Precompute RoPE frequency tables
	size_t head_dim = hidden_size / num_heads;
	auto [cos_table, sin_table] = precompute_rope_frequencies(
		head_dim, max_position_embeddings, rope_theta);
	cos_cache = cos_table;
	sin_cache = sin_table;
}

Variable LlamaModel::forward_ids(const std::vector<int>& token_ids, size_t position_offset) {
	using clock = std::chrono::high_resolution_clock;
	bool profiling = dcz::Config::get().profile;

	auto t0 = clock::now();

	// 1. Embed tokens
	Variable hidden_states = embed_tokens->forward_ids(token_ids);

	if (profiling) {
		auto t1 = clock::now();
		std::cerr << "[Profile] Embedding: "
				  << std::fixed << std::setprecision(2)
				  << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms" << std::endl;
	}

	// 2. Pass through decoder layers
	for (size_t i = 0; i < num_layers; ++i) {
		auto tl0 = clock::now();
		hidden_states = layers[i]->forward_with_cache(
			hidden_states, cos_cache, sin_cache, position_offset);
		if (profiling) {
			auto tl1 = clock::now();
			std::cerr << "[Profile] Layer " << i << ": "
					  << std::fixed << std::setprecision(2)
					  << std::chrono::duration<double, std::milli>(tl1 - tl0).count() << " ms" << std::endl;
		}
	}

	// 3. Final norm
	auto tn0 = clock::now();
	hidden_states = (*norm)(hidden_states);

	if (profiling) {
		auto tn1 = clock::now();
		std::cerr << "[Profile] Final norm: "
				  << std::fixed << std::setprecision(2)
				  << std::chrono::duration<double, std::milli>(tn1 - tn0).count() << " ms" << std::endl;
		std::cerr << "[Profile] LlamaModel total: "
				  << std::fixed << std::setprecision(2)
				  << std::chrono::duration<double, std::milli>(tn1 - t0).count() << " ms" << std::endl;
	}

	return hidden_states;
}

Variable LlamaModel::forward(const std::vector<Variable>& xs) {
	(void)xs;
	throw std::runtime_error("LlamaModel::forward not supported. Use forward_ids().");
}

void LlamaModel::reset_cache() {
	for (auto& layer : layers) {
		layer->reset_cache();
	}
}

void LlamaModel::load_from_npz(const cnpy::npz_t& npz, const std::string& prefix) {
	embed_tokens->load_from_npz(npz, prefix + ".embed_tokens");

	for (size_t i = 0; i < num_layers; ++i) {
		layers[i]->load_from_npz(npz, prefix + ".layers." + std::to_string(i));
	}

	norm->load_from_npz(npz, prefix + ".norm");
}

} // namespace layer


// ============================================================
// LlamaForCausalLM
// ============================================================

LlamaForCausalLM::LlamaForCausalLM(size_t vocab_size, size_t hidden_size,
									 size_t num_layers, size_t num_heads,
									 size_t num_kv_heads, size_t intermediate_size,
									 size_t max_position_embeddings,
									 float rope_theta, float rms_norm_eps)
{
	model = std::make_shared<layer::LlamaModel>(
		vocab_size, hidden_size, num_layers, num_heads, num_kv_heads,
		intermediate_size, max_position_embeddings, rope_theta, rms_norm_eps);

	register_sublayers("model", model);

	// Tie lm_head with embedding weights even before external weight loading.
	lm_head_weight = model->get_embed_tokens()->get_weight().transpose({1, 0}).contiguous();
}

Variable LlamaForCausalLM::forward_ids(const std::vector<int>& token_ids, size_t position_offset) {
	using clock = std::chrono::high_resolution_clock;
	bool profiling = dcz::Config::get().profile;

	auto t0 = clock::now();

	// 1. Get hidden states from model
	Variable hidden_states = model->forward_ids(token_ids, position_offset);

	auto t1 = clock::now();

	// Fallback for default-constructed/partially initialized states.
	if (lm_head_weight.empty()) {
		lm_head_weight = model->get_embed_tokens()->get_weight().transpose({1, 0}).contiguous();
	}

	// 2. lm_head: tied with embed_tokens weight (cached transposed)
	// Return logits for all provided positions: [batch, seq_len, vocab_size]

	Variable lm_weight(lm_head_weight);
	Variable logits = matmul(hidden_states, lm_weight);

	if (profiling) {
		auto t2 = clock::now();
		std::cerr << "[Profile] lm_head: "
				  << std::fixed << std::setprecision(2)
				  << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms" << std::endl;
		std::cerr << "[Profile] Total forward: "
				  << std::fixed << std::setprecision(2)
				  << std::chrono::duration<double, std::milli>(t2 - t0).count() << " ms" << std::endl;
	}

	return logits;  // [batch, seq_len, vocab_size]
}

Variable LlamaForCausalLM::forward(const std::vector<Variable>& xs) {
	(void)xs;
	throw std::runtime_error("LlamaForCausalLM::forward not supported. Use forward_ids().");
}

void LlamaForCausalLM::reset_cache() {
	model->reset_cache();
}

void LlamaForCausalLM::load_weights(const std::string& weights_path) {
	std::cout << "Loading Llama weights from: " << weights_path << std::endl;
	cnpy::npz_t npz = cnpy::npz_load(weights_path);
	std::cout << "NPZ loaded, " << npz.size() << " arrays" << std::endl;

	model->load_from_npz(npz, "model");

	// Cache transposed embedding weight for lm_head (avoid recomputing every forward)
	lm_head_weight = model->get_embed_tokens()->get_weight().transpose({1, 0}).contiguous();

	std::cout << "Weights loaded successfully." << std::endl;
}
