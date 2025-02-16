#include "deepczero.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <string>

using namespace tensor;
using namespace std::chrono;

// Timing utility
template<typename Func>
double measure_time_ms(Func&& func, int warmup = 3, int iterations = 100) {
    // Warmup
    for (int i = 0; i < warmup; ++i) {
        func();
    }

    // Actual measurement
    auto start = high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        func();
    }
    auto end = high_resolution_clock::now();

    double total_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
    return total_ms / iterations;
}

void print_header() {
    std::cout << "\n";
    std::cout << std::string(90, '=') << std::endl;
    std::cout << std::setw(15) << "Function"
              << std::setw(15) << "Size"
              << std::setw(15) << "MKL (ms)"
              << std::setw(15) << "Naive (ms)"
              << std::setw(15) << "Speedup"
              << std::setw(15) << "Status" << std::endl;
    std::cout << std::string(90, '=') << std::endl;
}

template<typename T>
void benchmark_function(const std::string& name,
                       std::function<Tensor<T>(const Tensor<T>&)> mkl_func,
                       std::function<Tensor<T>(const Tensor<T>&)> naive_func,
                       size_t size) {

    // Create test tensor
    Tensor<T> x({size});
    auto& x_data = x.raw_data();
    for (size_t i = 0; i < x_data.size(); ++i) {
        x_data[i] = static_cast<T>(1.0 + (rand() % 100) / 100.0);
    }

#ifdef USE_MKL
    // Benchmark MKL version
    Tensor<T> result_mkl;
    double time_mkl_ms = measure_time_ms([&]() {
        result_mkl = mkl_func(x);
    }, 3, 100);

    // Benchmark Naive version
    Tensor<T> result_naive;
    double time_naive_ms = measure_time_ms([&]() {
        result_naive = naive_func(x);
    }, 3, 100);

    double speedup = time_naive_ms / time_mkl_ms;

    // Verify correctness
    auto mkl_data = result_mkl.data();
    auto naive_data = result_naive.data();
    bool correct = true;
    const T epsilon = std::is_same<T, float>::value ? 1e-5 : 1e-10;

    for (size_t i = 0; i < std::min(size_t(1000), mkl_data.size()); ++i) {
        T diff = std::abs(mkl_data[i] - naive_data[i]);
        T rel_diff = diff / (std::abs(naive_data[i]) + epsilon);
        if (rel_diff > epsilon * 10) {
            correct = false;
            break;
        }
    }

    std::cout << std::setw(15) << name
              << std::setw(15) << size
              << std::setw(15) << std::fixed << std::setprecision(4) << time_mkl_ms
              << std::setw(15) << std::fixed << std::setprecision(4) << time_naive_ms
              << std::setw(14) << std::fixed << std::setprecision(2) << speedup << "x"
              << std::setw(15) << (correct ? "✓" : "✗") << std::endl;
#else
    std::cout << "MKL not enabled - skipping benchmark" << std::endl;
#endif
}

template<typename T>
void benchmark_pow_function(const std::string& name, size_t size) {
    // Create test tensor
    Tensor<T> x({size});
    auto& x_data = x.raw_data();
    for (size_t i = 0; i < x_data.size(); ++i) {
        x_data[i] = static_cast<T>(1.0 + (rand() % 100) / 100.0);
    }

    const T exponent = static_cast<T>(2.5);

#ifdef USE_MKL
    // Benchmark MKL version
    Tensor<T> result_mkl;
    double time_mkl_ms = measure_time_ms([&]() {
        result_mkl = pow_mkl(x, exponent);
    }, 3, 100);

    // Benchmark Naive version
    Tensor<T> result_naive;
    double time_naive_ms = measure_time_ms([&]() {
        result_naive = pow_naive(x, exponent);
    }, 3, 100);

    double speedup = time_naive_ms / time_mkl_ms;

    // Verify correctness
    auto mkl_data = result_mkl.data();
    auto naive_data = result_naive.data();
    bool correct = true;
    const T epsilon = std::is_same<T, float>::value ? 1e-5 : 1e-10;

    for (size_t i = 0; i < std::min(size_t(1000), mkl_data.size()); ++i) {
        T diff = std::abs(mkl_data[i] - naive_data[i]);
        T rel_diff = diff / (std::abs(naive_data[i]) + epsilon);
        if (rel_diff > epsilon * 10) {
            correct = false;
            break;
        }
    }

    std::cout << std::setw(15) << name
              << std::setw(15) << size
              << std::setw(15) << std::fixed << std::setprecision(4) << time_mkl_ms
              << std::setw(15) << std::fixed << std::setprecision(4) << time_naive_ms
              << std::setw(14) << std::fixed << std::setprecision(2) << speedup << "x"
              << std::setw(15) << (correct ? "✓" : "✗") << std::endl;
#else
    std::cout << "MKL not enabled - skipping benchmark" << std::endl;
#endif
}

void run_benchmarks() {
#ifndef USE_MKL
    std::cout << "\nERROR: This benchmark requires MKL to be enabled!" << std::endl;
    std::cout << "Please build with: make USE_MKL=1" << std::endl;
    return;
#else
    std::cout << "\n=== MKL VML vs Naive Math Functions Comparison ===" << std::endl;

    std::vector<size_t> sizes = {1000, 10000, 100000, 1000000};

    for (size_t size : sizes) {
        print_header();

        // Test exp
        benchmark_function<float>("exp",
            [](const Tensor<float>& x) { return exp_mkl(x); },
            [](const Tensor<float>& x) { return exp_naive(x); },
            size);

        // Test log
        benchmark_function<float>("log",
            [](const Tensor<float>& x) { return log_mkl(x); },
            [](const Tensor<float>& x) { return log_naive(x); },
            size);

        // Test sin
        benchmark_function<float>("sin",
            [](const Tensor<float>& x) { return sin_mkl(x); },
            [](const Tensor<float>& x) { return sin_naive(x); },
            size);

        // Test cos
        benchmark_function<float>("cos",
            [](const Tensor<float>& x) { return cos_mkl(x); },
            [](const Tensor<float>& x) { return cos_naive(x); },
            size);

        // Test tanh
        benchmark_function<float>("tanh",
            [](const Tensor<float>& x) { return tanh_mkl(x); },
            [](const Tensor<float>& x) { return tanh_naive(x); },
            size);

        // Test pow
        benchmark_pow_function<float>("pow", size);

        std::cout << std::string(90, '=') << std::endl;
    }
#endif
}

int main() {
    std::cout << "==================================================" << std::endl;
    std::cout << "      MKL VML Math Functions Benchmark           " << std::endl;
    std::cout << "==================================================" << std::endl;

    run_benchmarks();

    std::cout << "\n==================================================" << std::endl;
    std::cout << "              Benchmark Complete                  " << std::endl;
    std::cout << "==================================================" << std::endl;

    return 0;
}
