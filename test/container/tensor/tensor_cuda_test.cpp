#include "deepczero.hpp"
#include <iostream>
#include <cmath>

#ifndef USE_CUDA
int main() {
	std::cout << "CUDA not enabled. Build with: make USE_CUDA=1" << std::endl;
	return 0;
}
#else

using namespace tensor;

bool approx_equal(float a, float b, float tol = 1e-4f) {
	return std::abs(a - b) < tol;
}

// Compare two CPU tensors element-wise in logical (row-major) order
bool all_close(const Tensor<float>& expected, const Tensor<float>& actual, float tol = 1e-4f) {
	if (expected.get_shape() != actual.get_shape()) return false;
	auto e = expected.contiguous();
	auto a = actual.contiguous();
	for (size_t i = 0; i < e.size(); i++)
		if (!approx_equal(e.raw_data()[i], a.raw_data()[i], tol)) return false;
	return true;
}

void test_device_transfer() {
	std::cout << "=== Test: CPU <-> CUDA Transfer ===" << std::endl;

	Tensor<float> cpu_tensor({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

	// CPU -> CUDA
	auto gpu_tensor = cpu_tensor.to(dcz::cuda());
	std::cout << "  device: " << gpu_tensor.device() << std::endl;
	std::cout << "  is_device: " << gpu_tensor.is_device() << std::endl;
	std::cout << "  shape: [" << gpu_tensor.get_shape()[0] << ", " << gpu_tensor.get_shape()[1] << "]" << std::endl;
	std::cout << "  size: " << gpu_tensor.size() << std::endl;

	// CUDA -> CPU
	auto back = gpu_tensor.cpu();
	std::cout << "  roundtrip: ";
	for (size_t i = 0; i < back.size(); i++)
		std::cout << back.raw_data()[i] << " ";
	std::cout << std::endl;

	for (size_t i = 0; i < cpu_tensor.size(); i++) {
		if (!approx_equal(cpu_tensor.raw_data()[i], back.raw_data()[i])) {
			std::cout << "  FAIL: value mismatch at index " << i << std::endl;
			return;
		}
	}
	std::cout << "  PASS" << std::endl;
}

void test_elementwise_ops() {
	std::cout << "=== Test: Element-wise Ops on CUDA ===" << std::endl;

	Tensor<float> a_cpu({4}, {1.0f, 2.0f, 3.0f, 4.0f});
	Tensor<float> b_cpu({4}, {5.0f, 6.0f, 7.0f, 8.0f});

	auto a = a_cpu.to(dcz::cuda());
	auto b = b_cpu.to(dcz::cuda());

	// Add
	auto c = tensor::add(a, b).cpu();
	float expected_add[] = {6.0f, 8.0f, 10.0f, 12.0f};
	bool pass = true;
	for (size_t i = 0; i < 4; i++)
		if (!approx_equal(c.raw_data()[i], expected_add[i])) { pass = false; break; }
	std::cout << "  add: " << (pass ? "PASS" : "FAIL") << std::endl;

	// Sub
	auto d = tensor::sub(b, a).cpu();
	float expected_sub[] = {4.0f, 4.0f, 4.0f, 4.0f};
	pass = true;
	for (size_t i = 0; i < 4; i++)
		if (!approx_equal(d.raw_data()[i], expected_sub[i])) { pass = false; break; }
	std::cout << "  sub: " << (pass ? "PASS" : "FAIL") << std::endl;

	// Mul
	auto e = tensor::mul(a, b).cpu();
	float expected_mul[] = {5.0f, 12.0f, 21.0f, 32.0f};
	pass = true;
	for (size_t i = 0; i < 4; i++)
		if (!approx_equal(e.raw_data()[i], expected_mul[i])) { pass = false; break; }
	std::cout << "  mul: " << (pass ? "PASS" : "FAIL") << std::endl;

	// Div
	auto f = tensor::div(b, a).cpu();
	float expected_div[] = {5.0f, 3.0f, 7.0f/3.0f, 2.0f};
	pass = true;
	for (size_t i = 0; i < 4; i++)
		if (!approx_equal(f.raw_data()[i], expected_div[i])) { pass = false; break; }
	std::cout << "  div: " << (pass ? "PASS" : "FAIL") << std::endl;

	// Neg
	auto g = tensor::neg(a).cpu();
	float expected_neg[] = {-1.0f, -2.0f, -3.0f, -4.0f};
	pass = true;
	for (size_t i = 0; i < 4; i++)
		if (!approx_equal(g.raw_data()[i], expected_neg[i])) { pass = false; break; }
	std::cout << "  neg: " << (pass ? "PASS" : "FAIL") << std::endl;
}

void test_math_ops() {
	std::cout << "=== Test: Math Ops on CUDA ===" << std::endl;

	Tensor<float> x_cpu({4}, {0.5f, 1.0f, 1.5f, 2.0f});
	auto x = x_cpu.to(dcz::cuda());

	// Exp
	auto exp_result = tensor::exp(x).cpu();
	auto exp_expected = tensor::exp(x_cpu);
	bool pass = true;
	for (size_t i = 0; i < 4; i++)
		if (!approx_equal(exp_result.raw_data()[i], exp_expected.raw_data()[i])) { pass = false; break; }
	std::cout << "  exp: " << (pass ? "PASS" : "FAIL") << std::endl;

	// Tanh
	auto tanh_result = tensor::tanh(x).cpu();
	auto tanh_expected = tensor::tanh(x_cpu);
	pass = true;
	for (size_t i = 0; i < 4; i++)
		if (!approx_equal(tanh_result.raw_data()[i], tanh_expected.raw_data()[i])) { pass = false; break; }
	std::cout << "  tanh: " << (pass ? "PASS" : "FAIL") << std::endl;

	// Log
	auto log_result = tensor::log(x).cpu();
	auto log_expected = tensor::log(x_cpu);
	pass = true;
	for (size_t i = 0; i < 4; i++)
		if (!approx_equal(log_result.raw_data()[i], log_expected.raw_data()[i])) { pass = false; break; }
	std::cout << "  log: " << (pass ? "PASS" : "FAIL") << std::endl;
}

void test_gemm() {
	std::cout << "=== Test: GEMM (dot) on CUDA ===" << std::endl;

	Tensor<float> a_cpu({2, 3}, {1, 2, 3, 4, 5, 6});
	Tensor<float> b_cpu({3, 2}, {7, 8, 9, 10, 11, 12});

	auto a = a_cpu.to(dcz::cuda());
	auto b = b_cpu.to(dcz::cuda());

	auto c = tensor::dot(a, b).cpu();

	std::cout << "  result shape: [" << c.get_shape()[0] << ", " << c.get_shape()[1] << "]" << std::endl;

	float expected[] = {58.0f, 64.0f, 139.0f, 154.0f};
	bool pass = true;
	for (size_t i = 0; i < 4; i++) {
		if (!approx_equal(c.raw_data()[i], expected[i])) {
			pass = false;
			std::cout << "  mismatch at " << i << ": " << c.raw_data()[i] << " vs " << expected[i] << std::endl;
		}
	}
	std::cout << "  " << (pass ? "PASS" : "FAIL") << std::endl;
}

void test_reduction() {
	std::cout << "=== Test: Reductions on CUDA ===" << std::endl;

	Tensor<float> x_cpu({2, 3}, {1, 2, 3, 4, 5, 6});
	auto x = x_cpu.to(dcz::cuda());

	// Sum all
	auto sum_all = x.sum().cpu();
	bool pass = approx_equal(sum_all.raw_data()[0], 21.0f);
	std::cout << "  sum_all: " << sum_all.raw_data()[0] << " " << (pass ? "PASS" : "FAIL") << std::endl;

	// Sum along axis 1
	auto sum_1 = x.sum({1}).cpu();
	pass = approx_equal(sum_1.raw_data()[0], 6.0f) && approx_equal(sum_1.raw_data()[1], 15.0f);
	std::cout << "  sum axis=1: " << sum_1.raw_data()[0] << " " << sum_1.raw_data()[1]
			  << " " << (pass ? "PASS" : "FAIL") << std::endl;
}

void test_transpose() {
	std::cout << "=== Test: Transpose on CUDA (non-contiguous) ===" << std::endl;

	Tensor<float> a_cpu({2, 3}, {1, 2, 3, 4, 5, 6});
	Tensor<float> b_cpu({3, 2}, {1, 1, 2, 2, 3, 3});
	Tensor<float> c_cpu({2, 4}, {1, 2, 3, 4, 5, 6, 7, 8});

	auto a = a_cpu.to(dcz::cuda());
	auto b = b_cpu.to(dcz::cuda());
	auto c = c_cpu.to(dcz::cuda());

	// Transposed tensor read back to host
	bool pass = all_close(a_cpu.transpose(), a.transpose().cpu());
	std::cout << "  transpose: " << (pass ? "PASS" : "FAIL") << std::endl;

	// Element-wise op with a transposed operand
	pass = all_close(tensor::add(a_cpu.transpose(), b_cpu),
					 tensor::add(a.transpose(), b).cpu());
	std::cout << "  transpose + add: " << (pass ? "PASS" : "FAIL") << std::endl;

	// GEMM with a transposed operand: [3,2] @ [2,4]
	pass = all_close(tensor::dot(a_cpu.transpose(), c_cpu),
					 tensor::dot(a.transpose(), c).cpu());
	std::cout << "  dot(transpose(a), c): " << (pass ? "PASS" : "FAIL") << std::endl;

	// MatMul backward computes gy @ W^T and x^T @ gy internally
	Variable x_c(a_cpu), w_c(b_cpu);
	matmul(x_c, w_c).backward();
	Variable x_g(a), w_g(b);
	matmul(x_g, w_g).backward();
	pass = all_close(x_c.grad().data(), x_g.grad().data().cpu())
		&& all_close(w_c.grad().data(), w_g.grad().data().cpu());
	std::cout << "  matmul backward grads: " << (pass ? "PASS" : "FAIL") << std::endl;
}

void test_clone() {
	std::cout << "=== Test: Clone on CUDA (device-to-device deep copy) ===" << std::endl;

	Tensor<float> a_cpu({2, 2}, {1, 2, 3, 4});
	auto a = a_cpu.to(dcz::cuda());
	auto b = a.clone();

	bool pass = b.is_device() && b.device_buffer() != a.device_buffer()
		&& all_close(a_cpu, b.cpu());
	std::cout << "  clone values: " << (pass ? "PASS" : "FAIL") << std::endl;

	// Modifying the clone must not touch the original
	tensor::add_scalar_inplace(b, 10.0f);
	pass = all_close(a_cpu, a.cpu());
	std::cout << "  clone independence: " << (pass ? "PASS" : "FAIL") << std::endl;
}

Tensor<float> seq_tensor(const std::vector<size_t>& shape) {
	size_t n = 1;
	for (auto s : shape) n *= s;
	std::vector<float> data(n);
	for (size_t i = 0; i < n; i++) data[i] = static_cast<float>(i % 7) - 3.0f;
	return Tensor<float>(shape, data);
}

void test_batched_dot() {
	std::cout << "=== Test: Batched GEMM with broadcasting on CUDA ===" << std::endl;

	auto shape_str = [](const std::vector<size_t>& s) {
		std::string out = "[";
		for (size_t i = 0; i < s.size(); i++) out += (i ? "," : "") + std::to_string(s[i]);
		return out + "]";
	};

	std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>> cases = {
		{{3, 2, 3}, {3, 4}},
		{{2, 3}, {3, 3, 4}},
		{{3, 2, 3}, {3, 3, 4}},
		{{2, 1, 2, 3}, {1, 3, 3, 4}},
	};
	for (auto& [a_shape, b_shape] : cases) {
		auto a_cpu = seq_tensor(a_shape);
		auto b_cpu = seq_tensor(b_shape);
		bool pass = all_close(tensor::dot(a_cpu, b_cpu),
							  tensor::dot(a_cpu.to(dcz::cuda()), b_cpu.to(dcz::cuda())).cpu());
		std::cout << "  " << shape_str(a_shape) << " x " << shape_str(b_shape) << ": "
				  << (pass ? "PASS" : "FAIL") << std::endl;
	}
}

void test_device_mismatch() {
	std::cout << "=== Test: Device Mismatch Error ===" << std::endl;

	Tensor<float> cpu_a({4}, 1.0f);
	auto gpu_b = Tensor<float>({4}, 2.0f).to(dcz::cuda());

	try {
		auto c = tensor::add(cpu_a, gpu_b);
		std::cout << "  FAIL: should have thrown" << std::endl;
	} catch (const std::runtime_error& e) {
		std::cout << "  Caught: " << e.what() << std::endl;
		std::cout << "  PASS" << std::endl;
	}
}

int main() {
	std::cout << "CUDA GPU Backend Tests" << std::endl;
	std::cout << "======================" << std::endl;

	dcz::CUDAContext::get().print_device_info();
	std::cout << std::endl;

	test_device_transfer();
	test_elementwise_ops();
	test_math_ops();
	test_gemm();
	test_reduction();
	test_transpose();
	test_clone();
	test_batched_dot();
	test_device_mismatch();

	std::cout << "\nAll CUDA tests done." << std::endl;
	return 0;
}

#endif // USE_CUDA
