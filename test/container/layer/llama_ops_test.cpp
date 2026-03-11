#include "deepczero.hpp"
#include "utils/rope.hpp"

#include <iostream>
#include <cmath>
#include <cassert>

void test_rmsnorm() {
	std::cout << "=== Test RMSNorm ===" << std::endl;

	// Input: [1, 2, 4] (batch=1, seq=2, hidden=4)
	Tensor<> x_data({1, 2, 4}, {
		1.0f, 2.0f, 3.0f, 4.0f,    // position 0
		0.5f, 1.0f, 1.5f, 2.0f     // position 1
	});
	Variable x(x_data);

	// Weight: [4], all ones
	Tensor<> w_data({4}, {1.0f, 1.0f, 1.0f, 1.0f});
	Variable w(w_data);

	auto f = std::make_shared<function::RMSNorm>(1e-5f);
	Variable y = (*f)({x, w});

	auto shape = y.shape();
	std::cout << "Output shape: [" << shape[0] << ", " << shape[1] << ", " << shape[2] << "]" << std::endl;
	assert(shape[0] == 1 && shape[1] == 2 && shape[2] == 4);

	// Verify first position: x = [1, 2, 3, 4]
	// mean(x^2) = (1+4+9+16)/4 = 7.5
	// rsqrt(7.5 + 1e-5) = 1/sqrt(7.5) ≈ 0.3651
	// output = x * rsqrt * weight = [0.3651, 0.7303, 1.0954, 1.4606]
	const auto& out = y.data().raw_data();
	float expected_rsqrt = 1.0f / std::sqrt(7.5f + 1e-5f);
	float tol = 1e-4f;

	assert(std::abs(out[0] - 1.0f * expected_rsqrt) < tol);
	assert(std::abs(out[1] - 2.0f * expected_rsqrt) < tol);
	assert(std::abs(out[2] - 3.0f * expected_rsqrt) < tol);
	assert(std::abs(out[3] - 4.0f * expected_rsqrt) < tol);

	std::cout << "RMSNorm output[0:4]: " << out[0] << ", " << out[1] << ", " << out[2] << ", " << out[3] << std::endl;
	std::cout << "Expected rsqrt: " << expected_rsqrt << std::endl;
	std::cout << "RMSNorm test PASSED" << std::endl << std::endl;
}

void test_rmsnorm_with_weight() {
	std::cout << "=== Test RMSNorm with non-unit weight ===" << std::endl;

	Tensor<> x_data({1, 1, 4}, {1.0f, 2.0f, 3.0f, 4.0f});
	Variable x(x_data);

	Tensor<> w_data({4}, {0.5f, 1.0f, 2.0f, 0.1f});
	Variable w(w_data);

	auto f = std::make_shared<function::RMSNorm>(1e-5f);
	Variable y = (*f)({x, w});

	float expected_rsqrt = 1.0f / std::sqrt(7.5f + 1e-5f);
	const auto& out = y.data().raw_data();
	float tol = 1e-4f;

	assert(std::abs(out[0] - 1.0f * expected_rsqrt * 0.5f) < tol);
	assert(std::abs(out[1] - 2.0f * expected_rsqrt * 1.0f) < tol);
	assert(std::abs(out[2] - 3.0f * expected_rsqrt * 2.0f) < tol);
	assert(std::abs(out[3] - 4.0f * expected_rsqrt * 0.1f) < tol);

	std::cout << "RMSNorm with weight test PASSED" << std::endl << std::endl;
}

void test_embedding() {
	std::cout << "=== Test Embedding ===" << std::endl;

	layer::Embedding emb(5, 3);  // vocab=5, dim=3

	// Manually set weight
	Tensor<> w({5, 3}, {
		0.1f, 0.2f, 0.3f,   // token 0
		1.1f, 1.2f, 1.3f,   // token 1
		2.1f, 2.2f, 2.3f,   // token 2
		3.1f, 3.2f, 3.3f,   // token 3
		4.1f, 4.2f, 4.3f    // token 4
	});
	emb.set_param_data("W", w);

	// Lookup tokens [2, 0, 4]
	Variable y = emb.forward_ids({2, 0, 4});

	auto shape = y.shape();
	std::cout << "Output shape: [" << shape[0] << ", " << shape[1] << ", " << shape[2] << "]" << std::endl;
	assert(shape[0] == 1 && shape[1] == 3 && shape[2] == 3);

	const auto& out = y.data().raw_data();
	float tol = 1e-5f;

	// Token 2: [2.1, 2.2, 2.3]
	assert(std::abs(out[0] - 2.1f) < tol);
	assert(std::abs(out[1] - 2.2f) < tol);
	assert(std::abs(out[2] - 2.3f) < tol);

	// Token 0: [0.1, 0.2, 0.3]
	assert(std::abs(out[3] - 0.1f) < tol);
	assert(std::abs(out[4] - 0.2f) < tol);
	assert(std::abs(out[5] - 0.3f) < tol);

	// Token 4: [4.1, 4.2, 4.3]
	assert(std::abs(out[6] - 4.1f) < tol);
	assert(std::abs(out[7] - 4.2f) < tol);
	assert(std::abs(out[8] - 4.3f) < tol);

	std::cout << "Embedding test PASSED" << std::endl << std::endl;
}

void test_rope() {
	std::cout << "=== Test RoPE ===" << std::endl;

	size_t head_dim = 4;
	size_t max_seq = 8;
	float theta = 10000.0f;

	auto [cos_table, sin_table] = precompute_rope_frequencies(head_dim, max_seq, theta);

	std::cout << "RoPE cos_table shape: [" << cos_table.get_shape()[0] << ", " << cos_table.get_shape()[1] << "]" << std::endl;
	assert(cos_table.get_shape()[0] == max_seq);
	assert(cos_table.get_shape()[1] == head_dim);

	// Position 0: cos should be [1, 1, 1, 1], sin should be [0, 0, 0, 0]
	float tol = 1e-5f;
	assert(std::abs(cos_table.raw_data()[0] - 1.0f) < tol);
	assert(std::abs(cos_table.raw_data()[1] - 1.0f) < tol);
	assert(std::abs(sin_table.raw_data()[0] - 0.0f) < tol);
	assert(std::abs(sin_table.raw_data()[1] - 0.0f) < tol);

	std::cout << "RoPE position 0: cos=[" << cos_table.raw_data()[0] << ", " << cos_table.raw_data()[1] << "], sin=[" << sin_table.raw_data()[0] << ", " << sin_table.raw_data()[1] << "]" << std::endl;

	// Test apply_rope: identity at position 0 (sin=0, cos=1)
	// x: [1, 1, 2, 4] (batch=1, seq=1, heads=2, head_dim=4)
	Tensor<> x_data({1, 1, 2, 4}, {
		1.0f, 2.0f, 3.0f, 4.0f,
		5.0f, 6.0f, 7.0f, 8.0f
	});
	Variable x(x_data);

	Variable y = apply_rope(x, cos_table, sin_table, 0);
	const auto& out = y.data().raw_data();

	// At position 0, rotation is identity (cos=1, sin=0)
	assert(std::abs(out[0] - 1.0f) < tol);
	assert(std::abs(out[1] - 2.0f) < tol);
	assert(std::abs(out[2] - 3.0f) < tol);
	assert(std::abs(out[3] - 4.0f) < tol);

	std::cout << "RoPE at position 0 (identity): PASSED" << std::endl;

	// Test at position 1: should rotate
	Variable y1 = apply_rope(x, cos_table, sin_table, 1);
	const auto& out1 = y1.data().raw_data();

	// freq[0] = 1/theta^(0/4) = 1.0, freq[1] = 1/theta^(2/4) = 1/100
	// angle0 = 1 * 1.0 = 1.0, angle1 = 1 * 0.01 = 0.01
	float angle0 = 1.0f;
	float angle1 = 1.0f / 100.0f;
	float c0 = std::cos(angle0), s0 = std::sin(angle0);
	float c1 = std::cos(angle1), s1 = std::sin(angle1);

	// Pair (x[0], x[1]) rotated by angle0: (1*c0 - 2*s0, 1*s0 + 2*c0)
	float expected_0 = 1.0f * c0 - 2.0f * s0;
	float expected_1 = 1.0f * s0 + 2.0f * c0;
	// Pair (x[2], x[3]) rotated by angle1: (3*c1 - 4*s1, 3*s1 + 4*c1)
	float expected_2 = 3.0f * c1 - 4.0f * s1;
	float expected_3 = 3.0f * s1 + 4.0f * c1;

	std::cout << "RoPE at position 1: [" << out1[0] << ", " << out1[1] << ", " << out1[2] << ", " << out1[3] << "]" << std::endl;
	std::cout << "Expected:           [" << expected_0 << ", " << expected_1 << ", " << expected_2 << ", " << expected_3 << "]" << std::endl;

	assert(std::abs(out1[0] - expected_0) < tol);
	assert(std::abs(out1[1] - expected_1) < tol);
	assert(std::abs(out1[2] - expected_2) < tol);
	assert(std::abs(out1[3] - expected_3) < tol);

	std::cout << "RoPE at position 1 (rotation): PASSED" << std::endl;
	std::cout << "RoPE test PASSED" << std::endl << std::endl;
}

void test_llama_rmsnorm_layer() {
	std::cout << "=== Test LlamaRMSNorm Layer ===" << std::endl;

	layer::LlamaRMSNorm norm(4, 1e-5f);

	Tensor<> x_data({1, 2, 4}, {
		1.0f, 2.0f, 3.0f, 4.0f,
		0.5f, 1.0f, 1.5f, 2.0f
	});
	Variable x(x_data);

	Variable y = norm(x);

	auto shape = y.shape();
	assert(shape[0] == 1 && shape[1] == 2 && shape[2] == 4);

	// With weight=1.0, should match raw RMSNorm
	float rsqrt0 = 1.0f / std::sqrt(7.5f + 1e-5f);
	float tol = 1e-4f;
	assert(std::abs(y.data().raw_data()[0] - 1.0f * rsqrt0) < tol);

	std::cout << "LlamaRMSNorm layer test PASSED" << std::endl << std::endl;
}

int main() {
	dcz::UsingConfig no_grad("enable_backprop", false);

	test_rmsnorm();
	test_rmsnorm_with_weight();
	test_embedding();
	test_rope();
	test_llama_rmsnorm_layer();

	std::cout << "All Phase 1 tests PASSED!" << std::endl;
	return 0;
}
