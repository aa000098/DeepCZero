#include "deepczero.hpp"
#include <iostream>
#include <cmath>

#ifndef USE_SYCL
int main() {
	std::cout << "SYCL not enabled. Build with: make USE_SYCL=1" << std::endl;
	return 0;
}
#else

using namespace tensor;

bool approx_equal(float a, float b, float tol = 1e-5f) {
	return std::abs(a - b) < tol;
}

void test_device_transfer() {
	std::cout << "=== Test: CPU <-> SYCL Transfer ===" << std::endl;

	Tensor<float> cpu_tensor({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

	// CPU -> SYCL
	auto gpu_tensor = cpu_tensor.to(dcz::sycl());
	std::cout << "  device: " << gpu_tensor.device() << std::endl;
	std::cout << "  is_device: " << gpu_tensor.is_device() << std::endl;
	std::cout << "  shape: [" << gpu_tensor.get_shape()[0] << ", " << gpu_tensor.get_shape()[1] << "]" << std::endl;
	std::cout << "  size: " << gpu_tensor.size() << std::endl;

	// SYCL -> CPU
	auto back = gpu_tensor.cpu();
	std::cout << "  roundtrip: ";
	for (size_t i = 0; i < back.size(); i++)
		std::cout << back.raw_data()[i] << " ";
	std::cout << std::endl;

	// Verify values
	for (size_t i = 0; i < cpu_tensor.size(); i++) {
		if (!approx_equal(cpu_tensor.raw_data()[i], back.raw_data()[i])) {
			std::cout << "  FAIL: value mismatch at index " << i << std::endl;
			return;
		}
	}
	std::cout << "  PASS" << std::endl;
}

void test_elementwise_ops() {
	std::cout << "=== Test: Element-wise Ops on SYCL ===" << std::endl;

	Tensor<float> a_cpu({4}, {1.0f, 2.0f, 3.0f, 4.0f});
	Tensor<float> b_cpu({4}, {5.0f, 6.0f, 7.0f, 8.0f});

	auto a = a_cpu.to(dcz::sycl());
	auto b = b_cpu.to(dcz::sycl());

	// Add
	auto c = tensor::add(a, b);
	auto c_cpu = c.cpu();
	std::cout << "  add: ";
	for (size_t i = 0; i < c_cpu.size(); i++) std::cout << c_cpu.raw_data()[i] << " ";
	std::cout << std::endl;

	// Mul
	auto d = tensor::mul(a, b);
	auto d_cpu = d.cpu();
	std::cout << "  mul: ";
	for (size_t i = 0; i < d_cpu.size(); i++) std::cout << d_cpu.raw_data()[i] << " ";
	std::cout << std::endl;

	// Sub
	auto e = tensor::sub(b, a);
	auto e_cpu = e.cpu();
	std::cout << "  sub: ";
	for (size_t i = 0; i < e_cpu.size(); i++) std::cout << e_cpu.raw_data()[i] << " ";
	std::cout << std::endl;

	// Neg
	auto f = tensor::neg(a);
	auto f_cpu = f.cpu();
	std::cout << "  neg: ";
	for (size_t i = 0; i < f_cpu.size(); i++) std::cout << f_cpu.raw_data()[i] << " ";
	std::cout << std::endl;

	// Verify add
	bool pass = true;
	float expected_add[] = {6.0f, 8.0f, 10.0f, 12.0f};
	for (size_t i = 0; i < 4; i++) {
		if (!approx_equal(c_cpu.raw_data()[i], expected_add[i])) { pass = false; break; }
	}
	std::cout << "  " << (pass ? "PASS" : "FAIL") << std::endl;
}

void test_math_ops() {
	std::cout << "=== Test: Math Ops on SYCL ===" << std::endl;

	Tensor<float> x_cpu({4}, {0.5f, 1.0f, 1.5f, 2.0f});
	auto x = x_cpu.to(dcz::sycl());

	// Exp
	auto exp_result = tensor::exp(x).cpu();
	std::cout << "  exp: ";
	for (size_t i = 0; i < exp_result.size(); i++)
		std::cout << exp_result.raw_data()[i] << " ";
	std::cout << std::endl;

	// Tanh
	auto tanh_result = tensor::tanh(x).cpu();
	std::cout << "  tanh: ";
	for (size_t i = 0; i < tanh_result.size(); i++)
		std::cout << tanh_result.raw_data()[i] << " ";
	std::cout << std::endl;

	// Verify exp against CPU
	auto exp_cpu = tensor::exp(x_cpu);
	bool pass = true;
	for (size_t i = 0; i < 4; i++) {
		if (!approx_equal(exp_result.raw_data()[i], exp_cpu.raw_data()[i], 1e-4f)) {
			std::cout << "  exp mismatch at " << i << ": " << exp_result.raw_data()[i]
					  << " vs " << exp_cpu.raw_data()[i] << std::endl;
			pass = false;
		}
	}
	std::cout << "  " << (pass ? "PASS" : "FAIL") << std::endl;
}

void test_gemm() {
	std::cout << "=== Test: GEMM (dot) on SYCL ===" << std::endl;

	// 2x3 @ 3x2 = 2x2
	Tensor<float> a_cpu({2, 3}, {1, 2, 3, 4, 5, 6});
	Tensor<float> b_cpu({3, 2}, {7, 8, 9, 10, 11, 12});

	auto a = a_cpu.to(dcz::sycl());
	auto b = b_cpu.to(dcz::sycl());

	auto c = tensor::dot(a, b);
	auto c_cpu = c.cpu();

	std::cout << "  result shape: [" << c_cpu.get_shape()[0] << ", " << c_cpu.get_shape()[1] << "]" << std::endl;
	std::cout << "  values: ";
	for (size_t i = 0; i < c_cpu.size(); i++)
		std::cout << c_cpu.raw_data()[i] << " ";
	std::cout << std::endl;

	// Expected: [[58, 64], [139, 154]]
	float expected[] = {58.0f, 64.0f, 139.0f, 154.0f};
	bool pass = true;
	for (size_t i = 0; i < 4; i++) {
		if (!approx_equal(c_cpu.raw_data()[i], expected[i])) {
			pass = false;
			std::cout << "  mismatch at " << i << ": " << c_cpu.raw_data()[i] << " vs " << expected[i] << std::endl;
		}
	}
	std::cout << "  " << (pass ? "PASS" : "FAIL") << std::endl;
}

void test_device_mismatch() {
	std::cout << "=== Test: Device Mismatch Error ===" << std::endl;

	Tensor<float> cpu_a({4}, 1.0f);
	auto gpu_b = Tensor<float>({4}, 2.0f).to(dcz::sycl());

	try {
		auto c = tensor::add(cpu_a, gpu_b);
		std::cout << "  FAIL: should have thrown" << std::endl;
	} catch (const std::runtime_error& e) {
		std::cout << "  Caught: " << e.what() << std::endl;
		std::cout << "  PASS" << std::endl;
	}
}

int main() {
	std::cout << "SYCL GPU Backend Tests" << std::endl;
	std::cout << "======================" << std::endl;

	dcz::SYCLContext::get().print_device_info();
	std::cout << std::endl;

	test_device_transfer();
	test_elementwise_ops();
	test_math_ops();
	test_gemm();
	test_device_mismatch();

	std::cout << "\nAll SYCL tests done." << std::endl;
	return 0;
}

#endif // USE_SYCL
