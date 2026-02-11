#include "deepczero.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>

using namespace tensor;
using namespace std::chrono;

// Timing utility
template<typename Func>
double measure_time_ms(Func&& func, int warmup = 3, int iterations = 10) {
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
    return total_ms / iterations;  // Average time per iteration
}

void print_header() {
    std::cout << "\n";
    std::cout << std::string(100, '=') << std::endl;
    std::cout << std::setw(10) << "M"
              << std::setw(10) << "K"
              << std::setw(10) << "N"
              << std::setw(15) << "MKL (ms)"
              << std::setw(15) << "Naive (ms)"
              << std::setw(15) << "MKL GFLOPS"
              << std::setw(15) << "Naive GFLOPS"
              << std::setw(12) << "Speedup" << std::endl;
    std::cout << std::string(100, '=') << std::endl;
}

void benchmark_comparison() {
#ifndef USE_MKL
    std::cout << "\nERROR: This benchmark requires MKL to be enabled!" << std::endl;
    std::cout << "Please build with: make USE_MKL=1 bin/test/benchmark/matmul_mkl_vs_naive" << std::endl;
    return;
#else
    std::cout << "\n=== MKL vs Naive Implementation Comparison ===" << std::endl;
    std::cout << "Testing matrix multiplication: C = A @ B" << std::endl;

    print_header();

    // Test different matrix sizes
    std::vector<std::tuple<size_t, size_t, size_t>> sizes = {
        {64, 64, 64},
        {128, 128, 128},
        {256, 256, 256},
        {512, 512, 512},
        // {1024, 1024, 1024},
        // {2048, 2048, 2048},  // Uncomment if you want to test larger sizes
    };

    std::vector<double> speedups;

    for (const auto& [M, K, N] : sizes) {
        // Create random matrices
        Tensor<float> A({M, K});
        Tensor<float> B({K, N});

        // Initialize with random values
        auto& A_data = A.raw_data();
        auto& B_data = B.raw_data();
        for (size_t i = 0; i < A_data.size(); ++i) A_data[i] = static_cast<float>(rand()) / RAND_MAX;
        for (size_t i = 0; i < B_data.size(); ++i) B_data[i] = static_cast<float>(rand()) / RAND_MAX;

        // Benchmark MKL version
        Tensor<float> C_mkl;
        double time_mkl_ms = measure_time_ms([&]() {
            C_mkl = dot_mkl(A, B);
        }, 2, 5);

        // Benchmark Naive version
        Tensor<float> C_naive;
        double time_naive_ms = measure_time_ms([&]() {
            C_naive = dot_naive(A, B);
        }, 2, 5);

        // Calculate GFLOPS
        double flops = 2.0 * M * N * K;
        double gflops_mkl = flops / (time_mkl_ms * 1e6);
        double gflops_naive = flops / (time_naive_ms * 1e6);
        double speedup = time_naive_ms / time_mkl_ms;

        speedups.push_back(speedup);

        std::cout << std::setw(10) << M
                  << std::setw(10) << K
                  << std::setw(10) << N
                  << std::setw(15) << std::fixed << std::setprecision(2) << time_mkl_ms
                  << std::setw(15) << std::fixed << std::setprecision(2) << time_naive_ms
                  << std::setw(15) << std::fixed << std::setprecision(2) << gflops_mkl
                  << std::setw(15) << std::fixed << std::setprecision(2) << gflops_naive
                  << std::setw(11) << std::fixed << std::setprecision(2) << speedup << "x"
                  << std::endl;

        // Verify correctness (check a few random elements)
        auto mkl_data = C_mkl.data();
        auto naive_data = C_naive.data();
        bool correct = true;
        const float epsilon = 1e-3;  // Tolerance for floating point comparison

        for (size_t i = 0; i < std::min(size_t(100), mkl_data.size()); ++i) {
            if (std::abs(mkl_data[i] - naive_data[i]) > epsilon) {
                correct = false;
                std::cout << "  WARNING: Results differ at index " << i
                         << " (MKL: " << mkl_data[i] << ", Naive: " << naive_data[i] << ")" << std::endl;
                break;
            }
        }

        if (!correct) {
            std::cout << "  ERROR: MKL and Naive implementations produced different results!" << std::endl;
        }

        // Don't test too large matrices if they take too long
        if (time_naive_ms > 10000.0) {
            std::cout << "\n(Skipping larger matrices - naive version is too slow)" << std::endl;
            break;
        }
    }

    std::cout << std::string(100, '=') << std::endl;

    // Print summary statistics
    if (!speedups.empty()) {
        double avg_speedup = 0.0;
        for (double s : speedups) avg_speedup += s;
        avg_speedup /= speedups.size();

        double min_speedup = *std::min_element(speedups.begin(), speedups.end());
        double max_speedup = *std::max_element(speedups.begin(), speedups.end());

        std::cout << "\nSummary:" << std::endl;
        std::cout << "  Average Speedup: " << std::fixed << std::setprecision(2) << avg_speedup << "x" << std::endl;
        std::cout << "  Min Speedup:     " << std::fixed << std::setprecision(2) << min_speedup << "x" << std::endl;
        std::cout << "  Max Speedup:     " << std::fixed << std::setprecision(2) << max_speedup << "x" << std::endl;
    }
#endif
}

void benchmark_batched_comparison() {
#ifndef USE_MKL
    return;
#else
    std::cout << "\n\n=== Batched Matrix Multiplication Comparison ===" << std::endl;

    size_t batch = 4;
    size_t M = 256, K = 256, N = 256;

    std::cout << "Batch size: " << batch << ", Matrix size: " << M << "x" << K << " @ " << K << "x" << N << std::endl;

    Tensor<float> A({batch, M, K});
    Tensor<float> B({batch, K, N});

    auto& A_data = A.raw_data();
    auto& B_data = B.raw_data();
    for (size_t i = 0; i < A_data.size(); ++i) A_data[i] = static_cast<float>(rand()) / RAND_MAX;
    for (size_t i = 0; i < B_data.size(); ++i) B_data[i] = static_cast<float>(rand()) / RAND_MAX;

    // Benchmark MKL version
    Tensor<float> C_mkl;
    double time_mkl_ms = measure_time_ms([&]() {
        C_mkl = dot_mkl(A, B);
    }, 2, 5);

    // Benchmark Naive version
    Tensor<float> C_naive;
    double time_naive_ms = measure_time_ms([&]() {
        C_naive = dot_naive(A, B);
    }, 2, 5);

    double flops = 2.0 * batch * M * N * K;
    double gflops_mkl = flops / (time_mkl_ms * 1e6);
    double gflops_naive = flops / (time_naive_ms * 1e6);
    double speedup = time_naive_ms / time_mkl_ms;

    std::cout << "\nResults:" << std::endl;
    std::cout << "  MKL:   " << std::fixed << std::setprecision(2) << time_mkl_ms << " ms  ("
              << gflops_mkl << " GFLOPS)" << std::endl;
    std::cout << "  Naive: " << std::fixed << std::setprecision(2) << time_naive_ms << " ms  ("
              << gflops_naive << " GFLOPS)" << std::endl;
    std::cout << "  Speedup: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
#endif
}

int main() {
    std::cout << "==================================================" << std::endl;
    std::cout << "     MKL vs Naive Matrix Multiplication           " << std::endl;
    std::cout << "==================================================" << std::endl;

    benchmark_comparison();
    benchmark_batched_comparison();

    std::cout << "\n==================================================" << std::endl;
    std::cout << "              Benchmark Complete                  " << std::endl;
    std::cout << "==================================================" << std::endl;

    return 0;
}
