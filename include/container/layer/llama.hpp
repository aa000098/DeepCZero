#pragma once

#include "container/layer/layer.hpp"
#include "container/layer/model.hpp"

#include <vector>
#include <memory>
#include <string>
#include <cstddef>

namespace cnpy {
	struct NpyArray;
	using npz_t = std::map<std::string, NpyArray>;
}

namespace layer {

// ============================================================
// Embedding: token ID -> vector lookup
// ============================================================
class Embedding : public Layer {
private:
	size_t embed_dim;

public:
	Embedding() = default;
	Embedding(size_t vocab_size, size_t embed_dim);

	// Lookup by integer token IDs (not through autograd)
	// Returns Variable of shape [1, seq_len, embed_dim]
	Variable forward_ids(const std::vector<int>& token_ids);

	// Layer interface (unused for Embedding)
	Variable forward(const std::vector<Variable>& xs) override;

	void load_from_npz(const cnpy::npz_t& npz, const std::string& prefix);

	// Access weight for tied embeddings (lm_head)
	Tensor<> get_weight() const;
};

// ============================================================
// LlamaRMSNorm: RMS normalization layer
// ============================================================
class LlamaRMSNorm : public Layer {
private:
	float eps;

public:
	LlamaRMSNorm() = default;
	LlamaRMSNorm(size_t hidden_size, float eps = 1e-5f);

	Variable forward(const std::vector<Variable>& xs) override;

	void load_from_npz(const cnpy::npz_t& npz, const std::string& prefix);
};

// ============================================================
// LlamaAttention: Grouped Query Attention with KV Cache
// ============================================================
class LlamaAttention : public Layer {
private:
	size_t hidden_size;
	size_t num_heads;
	size_t num_kv_heads;
	size_t head_dim;
	size_t num_kv_groups;

	std::shared_ptr<Linear> q_proj;
	std::shared_ptr<Linear> k_proj;
	std::shared_ptr<Linear> v_proj;
	std::shared_ptr<Linear> o_proj;

	// KV Cache: pre-allocated tensors of [batch, max_seq_len, num_kv_heads, head_dim]
	Tensor<> k_cache;
	Tensor<> v_cache;
	size_t cache_len = 0;
	size_t cache_max_len = 0;  // allocated capacity (max_seq_len)

public:
	LlamaAttention() = default;
	LlamaAttention(size_t hidden_size, size_t num_heads, size_t num_kv_heads);

	// Forward with RoPE + KV cache
	Variable forward_attn(const Variable& hidden_states,
						  const Tensor<>& cos_cache,
						  const Tensor<>& sin_cache,
						  size_t position_offset);

	Variable forward(const std::vector<Variable>& xs) override;

	void reset_cache();
	void load_from_npz(const cnpy::npz_t& npz, const std::string& prefix);
};

// ============================================================
// LlamaMLP: SwiGLU Feed-Forward Network
// ============================================================
class LlamaMLP : public Layer {
private:
	std::shared_ptr<Linear> gate_proj;
	std::shared_ptr<Linear> up_proj;
	std::shared_ptr<Linear> down_proj;

public:
	LlamaMLP() = default;
	LlamaMLP(size_t hidden_size, size_t intermediate_size);

	Variable forward(const std::vector<Variable>& xs) override;

	void load_from_npz(const cnpy::npz_t& npz, const std::string& prefix);
};

// ============================================================
// LlamaDecoderLayer: one transformer block
// ============================================================
class LlamaDecoderLayer : public Layer {
private:
	std::shared_ptr<LlamaAttention> self_attn;
	std::shared_ptr<LlamaMLP> mlp;
	std::shared_ptr<LlamaRMSNorm> input_layernorm;
	std::shared_ptr<LlamaRMSNorm> post_attention_layernorm;

public:
	LlamaDecoderLayer() = default;
	LlamaDecoderLayer(size_t hidden_size, size_t num_heads,
					   size_t num_kv_heads, size_t intermediate_size,
					   float rms_norm_eps = 1e-5f);

	Variable forward_with_cache(const Variable& hidden_states,
								const Tensor<>& cos_cache,
								const Tensor<>& sin_cache,
								size_t position_offset);

	Variable forward(const std::vector<Variable>& xs) override;

	void reset_cache();
	void load_from_npz(const cnpy::npz_t& npz, const std::string& prefix);
};

// ============================================================
// LlamaModel: embeddings + decoder layers + final norm
// ============================================================
class LlamaModel : public Layer {
private:
	size_t num_layers;
	std::shared_ptr<Embedding> embed_tokens;
	std::vector<std::shared_ptr<LlamaDecoderLayer>> layers;
	std::shared_ptr<LlamaRMSNorm> norm;

	// Precomputed RoPE tables
	Tensor<> cos_cache;
	Tensor<> sin_cache;

public:
	LlamaModel() = default;
	LlamaModel(size_t vocab_size, size_t hidden_size, size_t num_layers,
			   size_t num_heads, size_t num_kv_heads, size_t intermediate_size,
			   size_t max_position_embeddings = 512,
			   float rope_theta = 500000.0f, float rms_norm_eps = 1e-5f);

	Variable forward_ids(const std::vector<int>& token_ids, size_t position_offset = 0);
	Variable forward(const std::vector<Variable>& xs) override;

	void reset_cache();
	void load_from_npz(const cnpy::npz_t& npz, const std::string& prefix);

	// Access embed_tokens for tied weights
	std::shared_ptr<Embedding> get_embed_tokens() const { return embed_tokens; }
};

} // namespace layer


// ============================================================
// LlamaForCausalLM: top-level model (outside namespace, like VGG16/YOLOv5)
// ============================================================
class LlamaForCausalLM : public Model {
private:
	std::shared_ptr<layer::LlamaModel> model;

	// Cached transposed embedding weight for lm_head (avoid recomputing every forward)
	Tensor<> lm_head_weight;  // [hidden_size, vocab_size]

public:
	LlamaForCausalLM() = default;
	LlamaForCausalLM(size_t vocab_size = 128256,
					  size_t hidden_size = 2048,
					  size_t num_layers = 16,
					  size_t num_heads = 32,
					  size_t num_kv_heads = 8,
					  size_t intermediate_size = 8192,
					  size_t max_position_embeddings = 512,
					  float rope_theta = 500000.0f,
					  float rms_norm_eps = 1e-5f);

	// Returns logits: Variable of shape [batch, seq_len, vocab_size]
	Variable forward_ids(const std::vector<int>& token_ids, size_t position_offset = 0);
	Variable forward(const std::vector<Variable>& xs) override;

	void to(const dcz::Device& device) override {
		Layer::to(device);
		if (!lm_head_weight.empty())
			lm_head_weight = lm_head_weight.to(device);
	}

	void reset_cache();
	void load_weights(const std::string& weights_path);
};
