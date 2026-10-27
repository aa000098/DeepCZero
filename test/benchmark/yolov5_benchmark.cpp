#include "deepczero.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>

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

static Tensor<> random_image(size_t size) {
	std::mt19937 gen(42);
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);
	Tensor<> x({1, 3, size, size});
	for (auto& v : x.raw_data()) v = dist(gen);
	return x;
}

static void bench_yolov5s(const dcz::Device& device, int iterations) {
	const size_t image_size = 640;

	YOLOv5 model(80, 0.33f, 0.50f);
	model.to(device);
	Variable x(random_image(image_size).to(device));

	auto start = Clock::now();
	model.forward_detect(x);
	sync(device);
	double first_ms = elapsed_ms(start);

	std::vector<double> times;
	for (int i = 0; i < iterations; ++i) {
		start = Clock::now();
		model.forward_detect(x);
		sync(device);
		times.push_back(elapsed_ms(start));
	}

	double mean_ms = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
	double min_ms = *std::min_element(times.begin(), times.end());

	std::cout << std::fixed << std::setprecision(2)
			  << "  " << std::setw(8) << std::left << device.str() << std::right
			  << "  first(+init): " << std::setw(10) << first_ms << " ms"
			  << "  mean: " << std::setw(9) << mean_ms << " ms"
			  << "  min: " << std::setw(9) << min_ms << " ms"
			  << "  FPS: " << std::setw(7) << 1000.0 / mean_ms
			  << "  (n=" << iterations << ")" << std::endl;
}

int main() {
	dcz::UsingConfig eval_mode("train", false);
	dcz::UsingConfig no_grad("enable_backprop", false);

	std::cout << "YOLOv5s Inference Benchmark" << std::endl;
	std::cout << "===========================" << std::endl;
	std::cout << "input [1, 3, 640, 640], random weights, forward_detect only (no decode/NMS)" << std::endl;

#ifdef USE_CUDA
	dcz::CUDAContext::get().print_device_info();
	bench_yolov5s(dcz::cuda(0), 5);
#endif

#ifdef USE_MKL
	bench_yolov5s(dcz::cpu(), 3);
#else
	std::cout << "  cpu     skipped (naive GEMM takes minutes per image; build with USE_MKL=1)" << std::endl;
#endif

	return 0;
}
