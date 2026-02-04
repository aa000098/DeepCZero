#include "deepczero.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <fstream>
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

struct BenchmarkResult {
    size_t M, K, N;
    double time_ms;
    double gflops;
    bool is_batched;
    size_t batch_size;
};

void save_results(const std::string& filename, const std::vector<BenchmarkResult>& results, bool use_mkl) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Failed to open output file: " << filename << std::endl;
        return;
    }

    outfile << "{\n";
    outfile << "  \"use_mkl\": " << (use_mkl ? "true" : "false") << ",\n";
    outfile << "  \"results\": [\n";

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        outfile << "    {\n";
        outfile << "      \"M\": " << r.M << ",\n";
        outfile << "      \"K\": " << r.K << ",\n";
        outfile << "      \"N\": " << r.N << ",\n";
        outfile << "      \"time_ms\": " << std::fixed << std::setprecision(4) << r.time_ms << ",\n";
        outfile << "      \"gflops\": " << std::fixed << std::setprecision(4) << r.gflops << ",\n";
        outfile << "      \"is_batched\": " << (r.is_batched ? "true" : "false");
        if (r.is_batched) {
            outfile << ",\n      \"batch_size\": " << r.batch_size;
        }
        outfile << "\n    }";
        if (i < results.size() - 1) {
            outfile << ",";
        }
        outfile << "\n";
    }

    outfile << "  ]\n";
    outfile << "}\n";
    outfile.close();

    std::cout << "Results saved to: " << filename << std::endl;
}

// Performance benchmark for different matrix sizes
std::vector<BenchmarkResult> run_benchmarks() {
    std::vector<BenchmarkResult> results;

    std::cout << "\n=== Performance Benchmark ===" << std::endl;

#ifdef USE_MKL
    std::cout << "MKL is ENABLED" << std::endl;
    bool use_mkl = true;
#else
    std::cout << "MKL is DISABLED (using naive implementation)" << std::endl;
    bool use_mkl = false;
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

        results.push_back({M, K, N, time_ms, gflops, false, 0});

        // Don't test too large matrices if they take too long
        if (time_ms > 5000.0) {
            std::cout << "(Skipping larger matrices due to time)" << std::endl;
            break;
        }
    }

    return results;
}

// Test batched matrix multiplication
void run_batched_benchmark(std::vector<BenchmarkResult>& results) {
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

    results.push_back({M, K, N, time_ms, gflops, true, batch});
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
