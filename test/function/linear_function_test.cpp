#include "deepczero.hpp"
#include <cassert>
#include <iostream>

void test_linear_forward_and_backward() {
	std::cout << "[Test] Linear forward & backward" << std::endl;

	// x: [2, 3], w: [3, 4], b: [4]
	Tensor<> x_data({2, 3}, {
		1, 2, 3,
		4, 5, 6
	});
	Tensor<> w_data({3, 4}, {
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0
	});
	Tensor<> b_data({4}, {1, 2, 3, 4});

	Variable x(x_data);
	Variable w(w_data);
	Variable b(b_data);

	Variable y = linear(x, w, b);

	// Expected: y = xW + b
	// xW = [
	//   [1,2,3,0],
	//   [4,5,6,0]
	// ]
	// y = xW + b = [
	//   [2,4,6,4],
	//   [5,7,9,4]
	// ]
	Tensor<> expected({2, 4}, {
		2, 4, 6, 4,
		5, 7, 9, 4
	});

	for (size_t i = 0; i < 8; ++i) {
		assert(std::abs(y.data().raw_data()[i] - expected.raw_data()[i]) < 1e-5);
	}

	// Backward test
	// Upstream gradient gy = ones([2, 4])
	Tensor<> gy_data({2, 4}, std::vector<float>(8, 1.0f));
	Variable gy(gy_data);
	y.set_grad(gy);
	y.backward();

	Variable gx = x.grad();
	Variable gw = w.grad();
	Variable gb = b.grad();

	// Expected gw = x^T @ gy
	// x^T = [ [1, 4],
	//         [2, 5],
	//         [3, 6] ]
	// gy   = [ [1, 1, 1, 1],
	//          [1, 1, 1, 1] ]
	// gw = x^T @ gy = [ [5,5,5,5], [7,7,7,7], [9,9,9,9] ]

	std::vector<float> gw_expected = {
		5, 5, 5, 5,
		7, 7, 7, 7,
		9, 9, 9, 9
	};

	for (size_t i = 0; i < gw_expected.size(); ++i) {
		assert(std::abs(gw.data().raw_data()[i] - gw_expected[i]) < 1e-4);
	}

	// gb = sum_to(gy, b.shape) = sum along axis 0 → [2, 2, 2, 2]
	std::vector<float> gb_expected = {2, 2, 2, 2};
	for (size_t i = 0; i < gb_expected.size(); ++i) {
		assert(std::abs(gb.data().raw_data()[i] - gb_expected[i]) < 1e-4);
	}

	std::cout << "✅ Linear forward & backward passed!" << std::endl;
}

void test_linear_backward_3d_input() {
	std::cout << "[Test] Linear / MatMul backward with 3D input" << std::endl;

	std::vector<float> xv(24), wv(20), bv(5);
	for (size_t i = 0; i < xv.size(); ++i) xv[i] = static_cast<float>(i % 7) - 3.0f;
	for (size_t i = 0; i < wv.size(); ++i) wv[i] = static_cast<float>(i % 5) * 0.5f - 1.0f;
	for (size_t i = 0; i < bv.size(); ++i) bv[i] = static_cast<float>(i);
	Tensor<> x_data({2, 3, 4}, xv);
	Tensor<> w_data({4, 5}, wv);
	Tensor<> b_data({5}, bv);

	auto check = [](const Variable& got, const Variable& ref, const std::vector<size_t>& shape) {
		assert(got.shape() == shape);
		auto g = got.data().contiguous();
		auto r = ref.data().contiguous();
		for (size_t i = 0; i < g.size(); ++i)
			assert(std::abs(g.raw_data()[i] - r.raw_data()[i]) < 1e-4);
	};

	Variable x3(x_data), w3(w_data), b3(b_data);
	linear(x3, w3, b3).backward();
	Variable x2(Tensor<>({6, 4}, xv)), w2(w_data), b2(b_data);
	linear(x2, w2, b2).backward();
	check(x3.grad(), x2.grad(), {2, 3, 4});
	check(w3.grad(), w2.grad(), {4, 5});
	check(b3.grad(), b2.grad(), {5});

	Variable mx3(x_data), mw3(w_data);
	matmul(mx3, mw3).backward();
	Variable mx2(Tensor<>({6, 4}, xv)), mw2(w_data);
	matmul(mx2, mw2).backward();
	check(mx3.grad(), mx2.grad(), {2, 3, 4});
	check(mw3.grad(), mw2.grad(), {4, 5});

	std::cout << "✅ Linear / MatMul backward with 3D input passed!" << std::endl;
}

int main() {
	test_linear_forward_and_backward();
	test_linear_backward_3d_input();
	return 0;
}

