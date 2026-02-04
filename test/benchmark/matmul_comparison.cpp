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

// Performance benchmark for different matrix sizes
void run_benchmarks() {

    std::cout << "\n=== Performance Benchmark ===" << std::endl;

#ifdef USE_MKL
    std::cout << "MKL is ENABLED" << std::endl;
#else
    std::cout << "MKL is DISABLED (using naive implementation)" << std::endl;
#endif

    std::cout << "\nMatrix multiplication: C = A @ B" << std::endl;
    std::cout << std::setw(10) << "M"
              << std::setw(10) << "K"
              << std::setw(10) << "N"
              << std::setw(15) << "Time (ms)"
              << std::setw(15) << "GFLOPS" << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    // Test different matrix sizes
    std::vector<std::tuple<size_t, size_t, size_t>> sizes = {
        {64, 64, 64},
        {128, 128, 128},
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
        {2048, 2048, 2048},
    };

    for (const auto& [M, K, N] : sizes) {
        // Create random matrices
        Tensor<float> A({M, K});
        Tensor<float> B({K, N});

        // Initialize with random values
        auto& A_data = A.raw_data();
        auto& B_data = B.raw_data();
        for (size_t i = 0; i < A_data.size(); ++i) A_data[i] = static_cast<float>(rand()) / RAND_MAX;
        for (size_t i = 0; i < B_data.size(); ++i) B_data[i] = static_cast<float>(rand()) / RAND_MAX;

        // Benchmark
        Tensor<float> C;
        double time_ms = measure_time_ms([&]() {
            C = dot(A, B);
        }, 2, 5);  // 2 warmup, 5 iterations

        // Calculate GFLOPS: (2*M*N*K) / (time_in_seconds * 10^9)
        double flops = 2.0 * M * N * K;
        double gflops = flops / (time_ms * 1e6);

        std::cout << std::setw(10) << M
                  << std::setw(10) << K
                  << std::setw(10) << N
                  << std::setw(15) << std::fixed << std::setprecision(2) << time_ms
                  << std::setw(15) << std::fixed << std::setprecision(2) << gflops
                  << std::endl;

        // Don't test too large matrices if they take too long
        if (time_ms > 5000.0) {
            std::cout << "(Skipping larger matrices due to time)" << std::endl;
            break;
        }
    }
}

// Test batched matrix multiplication
void run_batched_benchmark() {
    std::cout << "\n=== Batched Matrix Multiplication Test ===" << std::endl;

    size_t batch = 4;
    size_t M = 256, K = 256, N = 256;

    Tensor<float> A({batch, M, K});
    Tensor<float> B({batch, K, N});

    auto& A_data = A.raw_data();
    auto& B_data = B.raw_data();
    for (size_t i = 0; i < A_data.size(); ++i) A_data[i] = static_cast<float>(rand()) / RAND_MAX;
    for (size_t i = 0; i < B_data.size(); ++i) B_data[i] = static_cast<float>(rand()) / RAND_MAX;

    std::cout << "Batch size: " << batch << ", Matrix size: " << M << "x" << K << " @ " << K << "x" << N << std::endl;

    Tensor<float> C;
    double time_ms = measure_time_ms([&]() {
        C = dot(A, B);
    }, 2, 5);

    double flops = 2.0 * batch * M * N * K;
    double gflops = flops / (time_ms * 1e6);

    std::cout << "Time: " << std::fixed << std::setprecision(2) << time_ms << " ms" << std::endl;
    std::cout << "Performance: " << std::fixed << std::setprecision(2) << gflops << " GFLOPS" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "==================================================" << std::endl;
    std::cout << "  DeepCZero Matrix Multiplication Comparison     " << std::endl;
    std::cout << "==================================================" << std::endl;

#ifdef USE_MKL
    bool use_mkl = true;
    std::string output_file = "benchmark_mkl.json";
#else
    bool use_mkl = false;
    std::string output_file = "benchmark_no_mkl.json";
#endif

    // Allow custom output file from command line
    if (argc > 1) {
        output_file = argv[1];
    }

    std::vector<BenchmarkResult> results = run_benchmarks();
    run_batched_benchmark(results);

    save_results(output_file, results, use_mkl);

    std::cout << "\n==================================================" << std::endl;
    std::cout << "              Benchmark Complete                  " << std::endl;
    std::cout << "==================================================" << std::endl;

    return 0;
}
