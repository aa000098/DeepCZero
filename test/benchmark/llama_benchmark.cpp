#include "deepczero.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <string>

using namespace tensor;
using Clock = std::chrono::high_resolution_clock;

static double elapsed_ms(Clock::time_point start) {
	return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

static void sync(const dcz::Device& device) {
#ifdef USE_CUDA
	if (device.type == dcz::DeviceType::CUDA)
		CUDA_CHECK(cudaDeviceSynchronize());
#endif
	(void)device;
}

static void print_result(const dcz::Device& device, const std::string& test, size_t tokens, double ms) {
	std::cout << std::fixed << std::setprecision(2)
			  << "  " << std::setw(8) << std::left << device.str()
			  << std::setw(7) << test << std::right
			  << std::setw(10) << tokens / (ms / 1000.0) << " tok/s"
			  << std::setw(10) << ms / tokens << " ms/tok"
			  << std::setw(12) << ms << " ms total" << std::endl;
}

static void bench_llama_1b(const dcz::Device& device,
						   size_t pp_tokens, size_t tg_tokens, int pp_repeats) {
	auto start = Clock::now();
	LlamaForCausalLM model(128256, 2048, 16, 32, 8, 8192, 1024, 500000.0f, 1e-5f);
	model.to(device);
	model.forward_ids({1}, 0);
	sync(device);
	model.reset_cache();
	std::cout << std::fixed << std::setprecision(2)
			  << "  " << std::setw(8) << std::left << device.str() << std::right
			  << "init (random weights, one-off): " << elapsed_ms(start) / 1000.0 << " s" << std::endl;

	std::vector<int> prompt(pp_tokens);
	for (size_t i = 0; i < pp_tokens; ++i) prompt[i] = static_cast<int>(100 + i % 1000);

	double pp_best_ms = 0.0;
	for (int r = 0; r < pp_repeats; ++r) {
		model.reset_cache();
		start = Clock::now();
		model.forward_ids(prompt, 0);
		sync(device);
		double ms = elapsed_ms(start);
		if (r == 0 || ms < pp_best_ms) pp_best_ms = ms;
	}
	print_result(device, "pp" + std::to_string(pp_tokens), pp_tokens, pp_best_ms);

	model.reset_cache();
	start = Clock::now();
	for (size_t i = 0; i < tg_tokens; ++i)
		model.forward_ids({42}, i);
	sync(device);
	print_result(device, "tg" + std::to_string(tg_tokens), tg_tokens, elapsed_ms(start));

#ifdef USE_CUDA
	if (device.type == dcz::DeviceType::CUDA) {
		size_t free_bytes = 0, total_bytes = 0;
		CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
		std::cout << "  " << std::setw(8) << std::left << device.str() << std::right
				  << "GPU memory in use: " << (total_bytes - free_bytes) / (1024.0 * 1024.0)
				  << " MiB / " << total_bytes / (1024.0 * 1024.0) << " MiB" << std::endl;
	}
#endif
}

int main() {
	dcz::UsingConfig eval_mode("train", false);
	dcz::UsingConfig no_grad("enable_backprop", false);

	std::cout << "Llama 3.2 1B Benchmark (llama-bench style)" << std::endl;
	std::cout << "==========================================" << std::endl;
	std::cout << "fp32, random weights, pp = prompt processing, tg = token generation from empty cache" << std::endl;

#ifdef USE_CUDA
	dcz::CUDAContext::get().print_device_info();
	bench_llama_1b(dcz::cuda(0), 512, 128, 3);
#endif

#ifdef USE_MKL
	bench_llama_1b(dcz::cpu(), 512, 32, 1);
#else
	std::cout << "  cpu     skipped (naive GEMM is too slow for 1B params; build with USE_MKL=1)" << std::endl;
#endif

	return 0;
}
