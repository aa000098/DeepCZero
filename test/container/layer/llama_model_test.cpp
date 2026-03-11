#include "deepczero.hpp"

#include <iostream>
#include <cassert>
#include <cmath>

void test_llama_mlp() {
	std::cout << "=== Test LlamaMLP ===" << std::endl;

	size_t hidden = 8;
	size_t intermediate = 16;
	layer::LlamaMLP mlp(hidden, intermediate);
#ifdef USE_SYCL
	mlp.to(dcz::sycl());
#endif

	// Random-ish input: [1, 2, 8]
	std::vector<float> x_data(1 * 2 * hidden);
	for (size_t i = 0; i < x_data.size(); ++i)
		x_data[i] = static_cast<float>(i) * 0.1f - 0.4f;

	Variable x(Tensor<>({1, 2, hidden}, x_data));
#ifdef USE_SYCL
	x = x.to(dcz::sycl());
#endif
	Variable y = mlp(x);

	auto shape = y.shape();
	std::cout << "MLP output shape: [" << shape[0] << ", " << shape[1] << ", " << shape[2] << "]" << std::endl;
	assert(shape[0] == 1 && shape[1] == 2 && shape[2] == hidden);
	std::cout << "LlamaMLP test PASSED" << std::endl << std::endl;
}

void test_llama_attention() {
	std::cout << "=== Test LlamaAttention ===" << std::endl;

	size_t hidden = 64;
	size_t num_heads = 4;
	size_t num_kv_heads = 2;
	size_t head_dim = hidden / num_heads;  // 16

	layer::LlamaAttention attn(hidden, num_heads, num_kv_heads);
#ifdef USE_SYCL
	attn.to(dcz::sycl());
#endif

	// Precompute RoPE for small dim
	auto [cos_cache, sin_cache] = precompute_rope_frequencies(head_dim, 32, 500000.0f);

	// Input: [1, 3, 64]
	std::vector<float> x_data(1 * 3 * hidden);
	for (size_t i = 0; i < x_data.size(); ++i)
		x_data[i] = static_cast<float>(i % 7) * 0.1f - 0.3f;

	Variable x(Tensor<>({1, 3, hidden}, x_data));
#ifdef USE_SYCL
	x = x.to(dcz::sycl());
#endif

	// Forward
	Variable y = attn.forward_attn(x, cos_cache, sin_cache, 0);

	auto shape = y.shape();
	std::cout << "Attention output shape: [" << shape[0] << ", " << shape[1] << ", " << shape[2] << "]" << std::endl;
	assert(shape[0] == 1 && shape[1] == 3 && shape[2] == hidden);

	// Test KV cache: single token should work
	std::vector<float> x2_data(1 * 1 * hidden);
	for (size_t i = 0; i < x2_data.size(); ++i)
		x2_data[i] = static_cast<float>(i % 5) * 0.1f;

	Variable x2(Tensor<>({1, 1, hidden}, x2_data));
#ifdef USE_SYCL
	x2 = x2.to(dcz::sycl());
#endif
	Variable y2 = attn.forward_attn(x2, cos_cache, sin_cache, 3);

	auto shape2 = y2.shape();
	std::cout << "Attention with KV cache output shape: [" << shape2[0] << ", " << shape2[1] << ", " << shape2[2] << "]" << std::endl;
	assert(shape2[0] == 1 && shape2[1] == 1 && shape2[2] == hidden);

	std::cout << "LlamaAttention test PASSED" << std::endl << std::endl;
}

void test_llama_decoder_layer() {
	std::cout << "=== Test LlamaDecoderLayer ===" << std::endl;

	size_t hidden = 64;
	size_t num_heads = 4;
	size_t num_kv_heads = 2;
	size_t intermediate = 128;
	size_t head_dim = hidden / num_heads;

	layer::LlamaDecoderLayer decoder(hidden, num_heads, num_kv_heads, intermediate, 1e-5f);
#ifdef USE_SYCL
	decoder.to(dcz::sycl());
#endif

	auto [cos_cache, sin_cache] = precompute_rope_frequencies(head_dim, 32, 500000.0f);

	std::vector<float> x_data(1 * 3 * hidden);
	for (size_t i = 0; i < x_data.size(); ++i)
		x_data[i] = static_cast<float>(i % 11) * 0.05f - 0.25f;

	Variable x(Tensor<>({1, 3, hidden}, x_data));
#ifdef USE_SYCL
	x = x.to(dcz::sycl());
#endif
	Variable y = decoder.forward_with_cache(x, cos_cache, sin_cache, 0);

	auto shape = y.shape();
	std::cout << "Decoder output shape: [" << shape[0] << ", " << shape[1] << ", " << shape[2] << "]" << std::endl;
	assert(shape[0] == 1 && shape[1] == 3 && shape[2] == hidden);

	// Check output is not NaN
	auto y_cpu = y.data().is_device() ? y.data().cpu() : y.data();
	for (const auto& v : y_cpu.raw_data()) {
		assert(!std::isnan(v));
	}

	std::cout << "LlamaDecoderLayer test PASSED" << std::endl << std::endl;
}

void test_llama_model_small() {
	std::cout << "=== Test LlamaModel (small) ===" << std::endl;

	size_t vocab = 100;
	size_t hidden = 64;
	size_t num_layers = 2;
	size_t num_heads = 4;
	size_t num_kv_heads = 2;
	size_t intermediate = 128;

	layer::LlamaModel model(vocab, hidden, num_layers, num_heads, num_kv_heads,
							intermediate, 32, 500000.0f, 1e-5f);
#ifdef USE_SYCL
	model.to(dcz::sycl());
#endif

	// Forward with token IDs
	std::vector<int> token_ids = {5, 10, 15};
	Variable y = model.forward_ids(token_ids, 0);

	auto shape = y.shape();
	std::cout << "Model output shape: [" << shape[0] << ", " << shape[1] << ", " << shape[2] << "]" << std::endl;
	assert(shape[0] == 1 && shape[1] == 3 && shape[2] == hidden);

	// Check not NaN
	auto y_cpu2 = y.data().is_device() ? y.data().cpu() : y.data();
	for (const auto& v : y_cpu2.raw_data()) {
		assert(!std::isnan(v));
	}

	// Test KV cache with next token
	Variable y2 = model.forward_ids({20}, 3);
	auto shape2 = y2.shape();
	std::cout << "Model with cache output shape: [" << shape2[0] << ", " << shape2[1] << ", " << shape2[2] << "]" << std::endl;
	assert(shape2[0] == 1 && shape2[1] == 1 && shape2[2] == hidden);

	std::cout << "LlamaModel (small) test PASSED" << std::endl << std::endl;
}

void test_llama_causal_lm_small() {
	std::cout << "=== Test LlamaForCausalLM (small) ===" << std::endl;

	size_t vocab = 100;
	size_t hidden = 64;
	size_t num_layers = 2;
	size_t num_heads = 4;
	size_t num_kv_heads = 2;
	size_t intermediate = 128;

	LlamaForCausalLM model(vocab, hidden, num_layers, num_heads, num_kv_heads,
						   intermediate, 32, 500000.0f, 1e-5f);
#ifdef USE_SYCL
	model.to(dcz::sycl());
#endif

	// Forward
	std::vector<int> token_ids = {5, 10, 15};
	Variable logits = model.forward_ids(token_ids, 0);

	auto shape = logits.shape();
	std::cout << "Logits shape: [" << shape[0] << ", " << shape[1] << ", " << shape[2] << "]" << std::endl;
	assert(shape[0] == 1 && shape[1] == 3 && shape[2] == vocab);

	// Argmax of last position
	auto logits_cpu = logits.data().is_device() ? logits.data().cpu() : logits.data();
	const auto& data = logits_cpu.raw_data();
	size_t last_offset = 2 * vocab;
	float max_val = data[last_offset];
	int best = 0;
	for (size_t i = 1; i < vocab; ++i) {
		if (data[last_offset + i] > max_val) {
			max_val = data[last_offset + i];
			best = static_cast<int>(i);
		}
	}
	std::cout << "Next token (greedy): " << best << " (logit=" << max_val << ")" << std::endl;

	// Test KV cache step
	Variable logits2 = model.forward_ids({best}, 3);
	auto shape2 = logits2.shape();
	assert(shape2[0] == 1 && shape2[1] == 1 && shape2[2] == vocab);

	std::cout << "LlamaForCausalLM (small) test PASSED" << std::endl << std::endl;
}

int main() {
	dcz::UsingConfig eval_mode("train", false);
	dcz::UsingConfig no_grad("enable_backprop", false);

	test_llama_mlp();
	test_llama_attention();
	test_llama_decoder_layer();
	test_llama_model_small();
	test_llama_causal_lm_small();

	// Profile mode test
	{
		std::cout << "=== Test Profile Mode ===" << std::endl;
		dcz::UsingConfig profile_mode("profile", true);

		LlamaForCausalLM model(100, 64, 2, 4, 2, 128, 32, 500000.0f, 1e-5f);
#ifdef USE_SYCL
		model.to(dcz::sycl());
#endif
		Variable logits = model.forward_ids({5, 10, 15}, 0);
		std::cout << "Profile mode test PASSED" << std::endl << std::endl;
	}

	std::cout << "All model tests PASSED!" << std::endl;
	return 0;
}
