#include "deepczero.hpp"
#include <iostream>
#include <functional>
#include <cmath>

#ifndef USE_CUDA
int main() {
	std::cout << "CUDA not enabled. Build with: make USE_CUDA=1" << std::endl;
	return 0;
}
#else

using namespace tensor;

Tensor<float> seq_tensor(const std::vector<size_t>& shape, float scale = 0.1f) {
	size_t n = 1;
	for (auto s : shape) n *= s;
	std::vector<float> data(n);
	for (size_t i = 0; i < n; i++) data[i] = (static_cast<float>(i % 11) - 5.0f) * scale;
	return Tensor<float>(shape, data);
}

bool all_close(const Tensor<float>& expected, const Tensor<float>& actual, float tol = 1e-4f) {
	if (expected.get_shape() != actual.get_shape()) return false;
	auto e = expected.contiguous();
	auto a = actual.contiguous();
	for (size_t i = 0; i < e.size(); i++)
		if (std::abs(e.raw_data()[i] - a.raw_data()[i]) > tol) return false;
	return true;
}

void check(const std::string& name,
		   const std::vector<Tensor<float>>& inputs,
		   const std::function<Variable(const std::vector<Variable>&)>& fn) {
	std::vector<Variable> xc, xg;
	for (auto& t : inputs) {
		xc.emplace_back(t);
		xg.emplace_back(t.to(dcz::cuda()));
	}
	Variable yc = fn(xc);
	Variable yg = fn(xg);
	yc.backward();
	yg.backward();

	bool pass = yg.is_device() && all_close(yc.data(), yg.data().cpu());
	for (size_t i = 0; i < inputs.size(); i++) {
		if (!xc[i].has_grad()) continue;
		pass = pass && xg[i].has_grad() && xg[i].grad().data().is_device()
			&& all_close(xc[i].grad().data(), xg[i].grad().data().cpu());
	}
	std::cout << "  " << name << ": " << (pass ? "PASS" : "FAIL") << std::endl;
}

int main() {
	std::cout << "CUDA Function Tests (CPU vs GPU, forward + backward)" << std::endl;
	std::cout << "====================================================" << std::endl;

	Tensor<float> targets({4, 5}, 0.0f);
	for (size_t i = 0; i < targets.size(); i += 3) targets.raw_data()[i] = 1.0f;

	check("broadcast_to", {seq_tensor({1, 5})},
		  [](const std::vector<Variable>& x) { return broadcast_to(x[0], {4, 5}); });
	check("sum axis=1", {seq_tensor({4, 5})},
		  [](const std::vector<Variable>& x) { return sum(x[0], {1}); });
	check("x / scalar", {seq_tensor({4, 5})},
		  [](const std::vector<Variable>& x) { return x[0] / 2.0f; });
	check("binary_cross_entropy", {seq_tensor({4, 5}), targets},
		  [](const std::vector<Variable>& x) { return binary_cross_entropy(x[0], x[1]); });
	check("conv2d", {seq_tensor({2, 3, 6, 6}), seq_tensor({4, 3, 3, 3}, 0.05f), seq_tensor({4})},
		  [](const std::vector<Variable>& x) { return conv2d(x[0], x[1], x[2], {1, 1}, {1, 1}); });
	check("conv2d stride 2", {seq_tensor({2, 3, 7, 7}), seq_tensor({4, 3, 3, 3}, 0.05f), seq_tensor({4})},
		  [](const std::vector<Variable>& x) { return conv2d(x[0], x[1], x[2], {2, 2}, {0, 0}); });
	check("pooling 2x2", {seq_tensor({2, 3, 6, 6})},
		  [](const std::vector<Variable>& x) { return pooling(x[0], {2, 2}, {2, 2}, {0, 0}); });
	check("get_item", {seq_tensor({4, 5})},
		  [](const std::vector<Variable>& x) { return get_item(x[0], {1}); });

	std::cout << "\nAll CUDA function tests done." << std::endl;
	return 0;
}

#endif // USE_CUDA
