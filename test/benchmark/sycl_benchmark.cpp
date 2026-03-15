#include "deepczero.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <random>
#include <string>

#ifndef USE_SYCL
int main() {
	std::cout << "SYCL not enabled. Build with: make USE_SYCL=1" << std::endl;
	return 0;
}
#else

using namespace tensor;
using Clock = std::chrono::high_resolution_clock;

// Fill tensor with random data
Tensor<float> rand_tensor(const std::vector<size_t>& shape) {
	std::mt19937 gen(42);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
	size_t total = 1;
	for (auto s : shape) total *= s;
	std::vector<float> data(total);
	for (auto& v : data) v = dist(gen);
	return Tensor<float>(shape, data);
}

double measure_ms(std::function<void()> fn, int warmup = 1, int repeats = 5) {
	for (int i = 0; i < warmup; i++) fn();

	auto start = Clock::now();
	for (int i = 0; i < repeats; i++) fn();
	auto end = Clock::now();

	return std::chrono::duration<double, std::milli>(end - start).count() / repeats;
}

void print_row(const std::string& label, double cpu_ms, double gpu_ms) {
	double speedup = cpu_ms / gpu_ms;
	std::cout << "  " << std::setw(10) << std::left << label
			  << "CPU: " << std::setw(10) << std::right << std::fixed << std::setprecision(3) << cpu_ms << "ms"
			  << "  GPU: " << std::setw(10) << gpu_ms << "ms"
			  << "  (" << std::setprecision(1) << speedup << "x)" << std::endl;
}

void bench_gemm() {
	std::cout << "\n=== GEMM Benchmark (NxN @ NxN) ===" << std::endl;

	std::vector<size_t> sizes = {128, 256, 512, 1024, 2048};

	for (size_t N : sizes) {
		auto a_cpu = rand_tensor({N, N});
		auto b_cpu = rand_tensor({N, N});

		// Pre-transfer to GPU
		auto a_gpu = a_cpu.to(dcz::sycl());
		auto b_gpu = b_cpu.to(dcz::sycl());

		double cpu_ms = measure_ms([&]() {
			auto c = tensor::dot(a_cpu, b_cpu);
		});

		double gpu_ms = measure_ms([&]() {
			auto c = tensor::dot(a_gpu, b_gpu);
		});

		print_row("N=" + std::to_string(N), cpu_ms, gpu_ms);
	}
}

void bench_elementwise() {
	std::cout << "\n=== Element-wise Add Benchmark ===" << std::endl;

	std::vector<std::pair<std::string, size_t>> sizes = {
		{"100K", 100000}, {"1M", 1000000}, {"10M", 10000000}
	};

	for (auto& [label, N] : sizes) {
		auto a_cpu = rand_tensor({N});
		auto b_cpu = rand_tensor({N});

		auto a_gpu = a_cpu.to(dcz::sycl());
		auto b_gpu = b_cpu.to(dcz::sycl());

		double cpu_ms = measure_ms([&]() {
			auto c = tensor::add(a_cpu, b_cpu);
		});

		double gpu_ms = measure_ms([&]() {
			auto c = tensor::add(a_gpu, b_gpu);
		});

		print_row("N=" + label, cpu_ms, gpu_ms);
	}
}

void bench_exp() {
	std::cout << "\n=== Math Ops (exp) Benchmark ===" << std::endl;

	std::vector<std::pair<std::string, size_t>> sizes = {
		{"100K", 100000}, {"1M", 1000000}, {"10M", 10000000}
	};

	for (auto& [label, N] : sizes) {
		auto x_cpu = rand_tensor({N});
		auto x_gpu = x_cpu.to(dcz::sycl());

		double cpu_ms = measure_ms([&]() {
			auto r = tensor::exp(x_cpu);
		});

		double gpu_ms = measure_ms([&]() {
			auto r = tensor::exp(x_gpu);
		});

		print_row("N=" + label, cpu_ms, gpu_ms);
	}
}

void bench_transfer() {
	std::cout << "\n=== Host <-> Device Transfer Benchmark ===" << std::endl;

	std::vector<std::pair<std::string, size_t>> sizes = {
		{"1M", 1000000}, {"10M", 10000000}, {"50M", 50000000}
	};

	for (auto& [label, N] : sizes) {
		auto cpu_tensor = rand_tensor({N});

		double to_gpu_ms = measure_ms([&]() {
			auto g = cpu_tensor.to(dcz::sycl());
		});

		auto gpu_tensor = cpu_tensor.to(dcz::sycl());
		double to_cpu_ms = measure_ms([&]() {
			auto c = gpu_tensor.cpu();
		});

		size_t bytes = N * sizeof(float);
		double bw_to = (bytes / 1e6) / (to_gpu_ms / 1e3);
		double bw_from = (bytes / 1e6) / (to_cpu_ms / 1e3);

		std::cout << "  " << std::setw(10) << std::left << ("N=" + label)
				  << "to GPU: " << std::fixed << std::setprecision(3) << to_gpu_ms << "ms"
				  << " (" << std::setprecision(0) << bw_to << " MB/s)"
				  << "  to CPU: " << std::setprecision(3) << to_cpu_ms << "ms"
				  << " (" << std::setprecision(0) << bw_from << " MB/s)" << std::endl;
	}
}

int main() {
	std::cout << "SYCL GPU vs CPU Benchmark" << std::endl;
	std::cout << "=========================" << std::endl;

	dcz::SYCLContext::get().print_device_info();

	bench_transfer();
	bench_elementwise();
	bench_exp();
	bench_gemm();

	std::cout << "\nBenchmark done." << std::endl;
	return 0;
}

#endif // USE_SYCL
