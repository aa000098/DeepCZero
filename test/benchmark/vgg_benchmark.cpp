#include "deepczero.hpp"
#include "config/backend_config.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>

using namespace std::chrono;

template<typename Func>
double measure_time_ms(Func&& func, int warmup = 1, int iterations = 3) {
    for (int i = 0; i < warmup; ++i) {
        func();
    }

    auto start = high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        func();
    }
    auto end = high_resolution_clock::now();

    double total_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
    return total_ms / iterations;
}

void benchmark_vgg16_comparison() {
#ifndef USE_MKL
    std::cout << "\nERROR: This benchmark requires MKL to be enabled!" << std::endl;
    std::cout << "Please build with: make USE_MKL=1 bin/test/benchmark/vgg_benchmark" << std::endl;
    return;
#else
    std::cout << "\n=== VGG16 Forward Pass: MKL vs Naive ===" << std::endl;

    std::cout << std::string(100, '=') << std::endl;
    std::cout << std::setw(15) << "Batch"
              << std::setw(18) << "MKL (ms)"
              << std::setw(18) << "Naive (ms)"
              << std::setw(18) << "MKL (img/s)"
              << std::setw(18) << "Naive (img/s)"
              << std::setw(12) << "Speedup" << std::endl;
    std::cout << std::string(100, '=') << std::endl;

    std::vector<size_t> batch_sizes = {1, 2, 4};
    std::vector<double> speedups;

    for (size_t batch : batch_sizes) {
        Tensor<> x_data({batch, 3, 224, 224});
        auto& x_raw = x_data.raw_data();
        for (size_t i = 0; i < x_raw.size(); ++i) {
            x_raw[i] = static_cast<double>(rand() % 256) / 255.0;
        }
        Variable x(x_data);

        // MKL version
        dcz::BackendConfig::get().use_mkl = true;
        VGG16 model_mkl(false);
        double time_mkl = 0.0;

        try {
            time_mkl = measure_time_ms([&]() {
                dcz::UsingConfig test_mode("train", false);
                model_mkl.forward({x});
            }, 1, 3);
        } catch (const std::exception& e) {
            std::cerr << "MKL Error (batch=" << batch << "): " << e.what() << std::endl;
            continue;
        }

        // Naive version
        dcz::BackendConfig::get().use_mkl = false;
        VGG16 model_naive(false);
        double time_naive = 0.0;

        try {
            time_naive = measure_time_ms([&]() {
                dcz::UsingConfig test_mode("train", false);
                model_naive.forward({x});
            }, 1, 3);
        } catch (const std::exception& e) {
            std::cerr << "Naive Error (batch=" << batch << "): " << e.what() << std::endl;
            continue;
        }

        double throughput_mkl = (batch * 1000.0) / time_mkl;
        double throughput_naive = (batch * 1000.0) / time_naive;
        double speedup = time_naive / time_mkl;
        speedups.push_back(speedup);

        std::cout << std::setw(15) << batch
                  << std::setw(18) << std::fixed << std::setprecision(2) << time_mkl
                  << std::setw(18) << std::fixed << std::setprecision(2) << time_naive
                  << std::setw(17) << std::fixed << std::setprecision(2) << throughput_mkl << ""
                  << std::setw(17) << std::fixed << std::setprecision(2) << throughput_naive << ""
                  << std::setw(11) << std::fixed << std::setprecision(2) << speedup << "x"
                  << std::endl;

        if (time_naive > 30000.0) {
            std::cout << "\n(Skipping larger batches - naive is too slow)" << std::endl;
            break;
        }
    }

    // Restore default
    dcz::BackendConfig::get().use_mkl = true;

    std::cout << std::string(100, '=') << std::endl;

    if (!speedups.empty()) {
        double avg = 0.0;
        for (double s : speedups) avg += s;
        avg /= speedups.size();
        double min_s = *std::min_element(speedups.begin(), speedups.end());
        double max_s = *std::max_element(speedups.begin(), speedups.end());

        std::cout << "\nSummary:" << std::endl;
        std::cout << "  Average Speedup: " << std::fixed << std::setprecision(2) << avg << "x" << std::endl;
        std::cout << "  Min Speedup:     " << std::fixed << std::setprecision(2) << min_s << "x" << std::endl;
        std::cout << "  Max Speedup:     " << std::fixed << std::setprecision(2) << max_s << "x" << std::endl;
    }
#endif
}

int main() {
    srand(42);

    std::cout << "==================================================" << std::endl;
    std::cout << "     VGG16 Benchmark: MKL vs Naive                " << std::endl;
    std::cout << "==================================================" << std::endl;

    benchmark_vgg16_comparison();

    std::cout << "\n==================================================" << std::endl;
    std::cout << "              Benchmark Complete                  " << std::endl;
    std::cout << "==================================================" << std::endl;

    return 0;
}
