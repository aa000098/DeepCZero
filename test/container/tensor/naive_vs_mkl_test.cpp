// Cross-checks the naive CPU kernels against the MKL kernels on the cases the
// regular tests rarely hit: broadcasting, non-contiguous (transposed/sliced)
// inputs, and batched matmul.
#include "deepczero.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>

using namespace tensor;

static int failures = 0;

static Tensor<float> random_tensor(const std::vector<size_t>& shape) {
	Tensor<float> t(shape);
	auto& d = t.raw_data();
	for (auto& v : d) v = static_cast<float>(rand()) / RAND_MAX - 0.5f;
	return t;
}

static void expect_close(const std::string& name, const Tensor<float>& got,
                         const Tensor<float>& ref, float tol = 1e-4f) {
	if (got.get_shape() != ref.get_shape()) {
		std::cout << "[FAIL] " << name << ": shape mismatch" << std::endl;
		++failures;
		return;
	}
	auto g = got.data();
	auto r = ref.data();
	for (size_t i = 0; i < g.size(); ++i) {
		if (std::abs(g[i] - r[i]) > tol * (1.0f + std::abs(r[i]))) {
			std::cout << "[FAIL] " << name << " at " << i << ": " << g[i] << " vs " << r[i] << std::endl;
			++failures;
			return;
		}
	}
	std::cout << "[PASS] " << name << std::endl;
}

static void test_elementwise() {
	// Same shape
	auto a = random_tensor({4, 5, 6});
	auto b = random_tensor({4, 5, 6});
	expect_close("add same-shape", add_naive(a, b), add_mkl(a, b));
	expect_close("sub same-shape", sub_naive(a, b), sub_mkl(a, b));
	expect_close("mul same-shape", mul_naive(a, b), mul_mkl(a, b));
	for (auto& v : b.raw_data()) v += (v >= 0 ? 1.0f : -1.0f);  // keep away from zero
	expect_close("div same-shape", div_naive(a, b), div_mkl(a, b));

	// Broadcasting: reference is naive on pre-broadcast contiguous copies
	auto row = random_tensor({1, 6});
	auto col = random_tensor({5, 1});
	auto ref = add_mkl(broadcast_to(row, {5, 6}).contiguous(), broadcast_to(col, {5, 6}).contiguous());
	expect_close("add broadcast (1,6)+(5,1)", add_naive(row, col), ref);

	auto bias = random_tensor({6});
	auto ref2 = add_mkl(a, broadcast_to(bias, {4, 5, 6}).contiguous());
	expect_close("add broadcast (4,5,6)+(6)", add_naive(a, bias), ref2);

	// Non-contiguous operand (transposed view)
	auto m = random_tensor({7, 3});
	auto mt = m.transpose({1, 0});             // view, shape {3,7}
	auto n = random_tensor({3, 7});
	expect_close("mul transposed-view", mul_naive(mt, n), mul_mkl(mt.contiguous(), n));

	// Scalar-like 0-d / 1-element broadcast
	auto s = random_tensor({1});
	auto ref3 = mul_mkl(a, broadcast_to(s, {4, 5, 6}).contiguous());
	expect_close("mul broadcast scalar", mul_naive(a, s), ref3);
}

static void test_dot() {
	auto A = random_tensor({33, 17});
	auto B = random_tensor({17, 29});
	expect_close("dot 2D", dot_naive(A, B), dot_mkl(A, B));

	auto At = random_tensor({17, 33}).transpose({1, 0});  // {33,17} view
	expect_close("dot transposed A", dot_naive(At, B), dot_mkl(At, B));

	auto Ab = random_tensor({3, 2, 8, 5});
	auto Bb = random_tensor({3, 2, 5, 9});
	expect_close("dot batched 4D", dot_naive(Ab, Bb), dot_mkl(Ab, Bb));

	auto B1 = random_tensor({5, 9});
	expect_close("dot batch-broadcast B", dot_naive(Ab, B1), dot_mkl(Ab, B1));

	auto Ab1 = random_tensor({1, 8, 5});
	auto Bb2 = random_tensor({4, 5, 9});
	expect_close("dot batch-broadcast A", dot_naive(Ab1, Bb2), dot_mkl(Ab1, Bb2));
}

int main() {
#ifndef USE_MKL
	std::cout << "MKL disabled; nothing to cross-check." << std::endl;
	return 0;
#else
	srand(0);
	test_elementwise();
	test_dot();
	if (failures) {
		std::cout << failures << " failure(s)" << std::endl;
		return 1;
	}
	std::cout << "All naive-vs-MKL checks passed" << std::endl;
	return 0;
#endif
}
