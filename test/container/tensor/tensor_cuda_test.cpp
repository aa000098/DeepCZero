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
	test_device_mismatch();

	std::cout << "\nAll CUDA tests done." << std::endl;
	return 0;
}

#endif // USE_CUDA
