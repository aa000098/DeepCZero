#include "deepczero.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <cmath>

#ifdef USE_DNNL
#include "container/tensor/tensor_ops_dnnl.hpp"
#endif

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

// Check if two tensors are approximately equal
template<typename T>
bool tensors_close(const Tensor<T>& a, const Tensor<T>& b, T rtol = 1e-3, T atol = 1e-4) {
    if (a.get_shape() != b.get_shape()) return false;

    const auto& a_data = a.raw_data();
    const auto& b_data = b.raw_data();

    for (size_t i = 0; i < a_data.size(); ++i) {
        T diff = std::abs(a_data[i] - b_data[i]);
        T threshold = atol + rtol * std::abs(b_data[i]);
        if (diff > threshold) {
            std::cout << "Mismatch at index " << i << ": "
                      << a_data[i] << " vs " << b_data[i]
                      << " (diff: " << diff << ")" << std::endl;
            return false;
        }
    }
    return true;
}

struct ConvConfig {
    size_t N, C, H, W;        // Input: [N, C, H, W]
    size_t OC, KH, KW;        // Weight: [OC, C, KH, KW]
    size_t stride_h, stride_w;
    size_t pad_h, pad_w;
    std::string name;
};

void benchmark_conv2d(const ConvConfig& config) {
    std::cout << "\n" << config.name << std::endl;
    std::cout << "  Input: [" << config.N << ", " << config.C << ", "
              << config.H << ", " << config.W << "]" << std::endl;
    std::cout << "  Weight: [" << config.OC << ", " << config.C << ", "
              << config.KH << ", " << config.KW << "]" << std::endl;
    std::cout << "  Stride: [" << config.stride_h << ", " << config.stride_w << "]"
              << ", Pad: [" << config.pad_h << ", " << config.pad_w << "]" << std::endl;

    // Calculate output size
    size_t OH = (config.H + 2 * config.pad_h - config.KH) / config.stride_h + 1;
    size_t OW = (config.W + 2 * config.pad_w - config.KW) / config.stride_w + 1;

    // Create test data
    Tensor<float> x({config.N, config.C, config.H, config.W});
    Tensor<float> w({config.OC, config.C, config.KH, config.KW});
    Tensor<float> b({config.OC});

    // Initialize with small random values
    auto& x_data = x.raw_data();
    auto& w_data = w.raw_data();
    auto& b_data = b.raw_data();

    for (size_t i = 0; i < x_data.size(); ++i)
        x_data[i] = (static_cast<float>(rand()) / RAND_MAX) * 0.1f;
    for (size_t i = 0; i < w_data.size(); ++i)
        w_data[i] = (static_cast<float>(rand()) / RAND_MAX) * 0.1f;
    for (size_t i = 0; i < b_data.size(); ++i)
        b_data[i] = (static_cast<float>(rand()) / RAND_MAX) * 0.1f;

    // Calculate FLOPs: N * OC * OH * OW * (2 * C * KH * KW)
    double flops = static_cast<double>(config.N) * config.OC * OH * OW *
                   (2.0 * config.C * config.KH * config.KW);

    // Benchmark im2col method
    Tensor<float> y_im2col;
    double time_im2col = measure_time_ms([&]() {
        Tensor<float> col = im2col_array(x, {config.KH, config.KW},
                                          {config.stride_h, config.stride_w},
                                          {config.pad_h, config.pad_w}, false);
        Tensor<float> y = tensordot(col, w, {{1,2,3}, {1,2,3}});
        y += b;
        y_im2col = y.transpose({0, 3, 1, 2});
    }, 2, 5);

    double gflops_im2col = flops / (time_im2col * 1e6);

    std::cout << "\n  [im2col]:" << std::endl;
    std::cout << "    Time: " << std::fixed << std::setprecision(2)
              << time_im2col << " ms" << std::endl;
    std::cout << "    GFLOPS: " << std::fixed << std::setprecision(2)
              << gflops_im2col << std::endl;

#ifdef USE_DNNL
    // Benchmark oneDNN method
    Tensor<float> y_dnnl;
    double time_dnnl = measure_time_ms([&]() {
        y_dnnl = conv2d_dnnl(x, w, b,
                             {config.stride_h, config.stride_w},
                             {config.pad_h, config.pad_w});
    }, 2, 5);

    double gflops_dnnl = flops / (time_dnnl * 1e6);
    double speedup = time_im2col / time_dnnl;

    std::cout << "\n  [oneDNN]:" << std::endl;
    std::cout << "    Time: " << std::fixed << std::setprecision(2)
              << time_dnnl << " ms" << std::endl;
    std::cout << "    GFLOPS: " << std::fixed << std::setprecision(2)
              << gflops_dnnl << std::endl;
    std::cout << "    Speedup: " << std::fixed << std::setprecision(2)
              << speedup << "x" << std::endl;

    // Correctness check
    std::cout << "\n  [Correctness]:" << std::endl;
    bool correct = tensors_close(y_im2col, y_dnnl, 1e-3f, 1e-3f);
    if (correct) {
        std::cout << "    ✓ Results match!" << std::endl;
    } else {
        std::cout << "    ✗ Results differ!" << std::endl;
    }
#else
    std::cout << "\n  [oneDNN]: NOT ENABLED" << std::endl;
    std::cout << "  Build with USE_DNNL=1 to enable oneDNN" << std::endl;
#endif

    std::cout << std::string(70, '-') << std::endl;
}

int main() {
    std::cout << "==================================================" << std::endl;
    std::cout << "   DeepCZero Conv2D Benchmark: im2col vs oneDNN  " << std::endl;
    std::cout << "==================================================" << std::endl;

#ifdef USE_MKL
    std::cout << "[MKL] ENABLED" << std::endl;
#else
    std::cout << "[MKL] DISABLED" << std::endl;
#endif

#ifdef USE_DNNL
    std::cout << "[DNNL] ENABLED" << std::endl;
#else
    std::cout << "[DNNL] DISABLED (build with USE_DNNL=1 to enable)" << std::endl;
#endif

    // Test configurations
    std::vector<ConvConfig> configs = {
        // Small batch, typical CNN sizes
        {1, 3, 224, 224, 64, 3, 3, 1, 1, 1, 1, "VGG-like (small batch)"},
        {1, 64, 56, 56, 128, 3, 3, 1, 1, 1, 1, "VGG-like (mid layer)"},
        {1, 512, 14, 14, 512, 3, 3, 1, 1, 1, 1, "VGG-like (deep layer)"},

        // Larger batch
        {8, 3, 224, 224, 64, 3, 3, 1, 1, 1, 1, "Batch=8, 3x3 conv"},
        {16, 64, 56, 56, 128, 3, 3, 1, 1, 1, 1, "Batch=16, 3x3 conv"},

        // Different kernel sizes
        {4, 64, 112, 112, 64, 1, 1, 1, 1, 0, 0, "1x1 conv (pointwise)"},
        {4, 64, 56, 56, 128, 5, 5, 1, 1, 2, 2, "5x5 conv"},

        // Stride > 1
        {4, 64, 112, 112, 128, 3, 3, 2, 2, 1, 1, "Stride=2 (downsampling)"},
    };

    for (const auto& config : configs) {
        benchmark_conv2d(config);
    }

    std::cout << "\n==================================================" << std::endl;
    std::cout << "              Benchmark Complete                  " << std::endl;
    std::cout << "==================================================" << std::endl;

    return 0;
}
