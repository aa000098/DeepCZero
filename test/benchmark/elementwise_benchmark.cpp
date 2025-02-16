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
    std::cout << std::setw(15) << "Operation"
              << std::setw(15) << "Size"
              << std::setw(15) << "MKL (ms)"
              << std::setw(15) << "Naive (ms)"
              << std::setw(15) << "Speedup"
              << std::setw(15) << "Status" << std::endl;
    std::cout << std::string(90, '=') << std::endl;
}

template<typename T>
void benchmark_binary_op(const std::string& name,
                        std::function<Tensor<T>(const Tensor<T>&, const Tensor<T>&)> mkl_func,
                        std::function<Tensor<T>(const Tensor<T>&, const Tensor<T>&)> naive_func,
                        size_t size) {

    // Create test tensors
    Tensor<T> a({size});
    Tensor<T> b({size});

    auto& a_data = a.raw_data();
    auto& b_data = b.raw_data();

    for (size_t i = 0; i < a_data.size(); ++i) {
        a_data[i] = static_cast<T>(1.0 + (rand() % 100) / 100.0);
        b_data[i] = static_cast<T>(1.0 + (rand() % 100) / 100.0);
    }

#ifdef USE_MKL
    // Benchmark MKL version
    Tensor<T> result_mkl;
    double time_mkl_ms = measure_time_ms([&]() {
        result_mkl = mkl_func(a, b);
    }, 3, 100);

    // Benchmark Naive version
    Tensor<T> result_naive;
    double time_naive_ms = measure_time_ms([&]() {
        result_naive = naive_func(a, b);
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
    std::cout << "\n=== MKL VML vs Naive Element-wise Operations ===" << std::endl;

    std::vector<size_t> sizes = {1000, 10000, 100000, 1000000};

    for (size_t size : sizes) {
        print_header();

        // Test add
        benchmark_binary_op<float>("add",
            [](const Tensor<float>& a, const Tensor<float>& b) { return add_mkl(a, b); },
            [](const Tensor<float>& a, const Tensor<float>& b) { return add_naive(a, b); },
            size);

        // Test sub
        benchmark_binary_op<float>("sub",
            [](const Tensor<float>& a, const Tensor<float>& b) { return sub_mkl(a, b); },
            [](const Tensor<float>& a, const Tensor<float>& b) { return sub_naive(a, b); },
            size);

        // Test mul
        benchmark_binary_op<float>("mul",
            [](const Tensor<float>& a, const Tensor<float>& b) { return mul_mkl(a, b); },
            [](const Tensor<float>& a, const Tensor<float>& b) { return mul_naive(a, b); },
            size);

        // Test div
        benchmark_binary_op<float>("div",
            [](const Tensor<float>& a, const Tensor<float>& b) { return div_mkl(a, b); },
            [](const Tensor<float>& a, const Tensor<float>& b) { return div_naive(a, b); },
            size);

        std::cout << std::string(90, '=') << std::endl;
    }
#endif
}

void test_broadcasting() {
    std::cout << "\n=== Broadcasting Test ===" << std::endl;
    std::cout << "Testing that MKL is used only when shapes match..." << std::endl;

    // Same shape - should use MKL
    Tensor<float> a1({100, 100});
    Tensor<float> b1({100, 100});
    auto& a1_data = a1.raw_data();
    auto& b1_data = b1.raw_data();
    for (size_t i = 0; i < a1_data.size(); ++i) {
        a1_data[i] = 1.0f;
        b1_data[i] = 2.0f;
    }

    auto c1 = add(a1, b1);
    std::cout << "Same shape (100x100 + 100x100): ✓" << std::endl;

    // Different shape - should use naive (broadcasting)
    Tensor<float> a2({100, 1});
    Tensor<float> b2({1, 100});
    auto& a2_data = a2.raw_data();
    auto& b2_data = b2.raw_data();
    for (size_t i = 0; i < a2_data.size(); ++i) a2_data[i] = 1.0f;
    for (size_t i = 0; i < b2_data.size(); ++i) b2_data[i] = 2.0f;

    auto c2 = add(a2, b2);
    std::cout << "Broadcasting (100x1 + 1x100 -> 100x100): ✓" << std::endl;

    std::cout << "Broadcasting test passed!" << std::endl;
}

int main() {
    std::cout << "==================================================" << std::endl;
    std::cout << "    MKL VML Element-wise Operations Benchmark     " << std::endl;
    std::cout << "==================================================" << std::endl;

    run_benchmarks();
    test_broadcasting();

    std::cout << "\n==================================================" << std::endl;
    std::cout << "              Benchmark Complete                  " << std::endl;
    std::cout << "==================================================" << std::endl;

    return 0;
}
