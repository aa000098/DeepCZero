#!/usr/bin/env python3
"""PyTorch CPU reference timings for the shapes used by the C++ benchmarks here.

Run next to the DeepCZero benchmarks to compare against an established runtime
instead of the in-tree naive kernels:

    ./bin/benchmark/matmul_mkl_vs_naive
    ./bin/benchmark/elementwise_benchmark
    ./bin/benchmark/vgg_benchmark
    python3 test/benchmark/torch_benchmark.py            # all sections
    python3 test/benchmark/torch_benchmark.py matmul     # one section

Timings use the same warmup/iteration counts as the C++ benchmarks.
This file is not picked up by `make bench` (it only runs the .cpp benchmarks).
"""
import sys
import time

import torch

torch.set_grad_enabled(False)


def measure_ms(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) * 1000.0 / iters


def bench_matmul():
    print("\n=== PyTorch matmul (float32, CPU) ===")
    print(f"{'M':>10}{'K':>10}{'N':>10}{'Time (ms)':>15}{'GFLOPS':>15}")
    for m, k, n in [(64, 64, 64), (128, 128, 128), (256, 256, 256),
                    (512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)]:
        a = torch.rand(m, k)
        b = torch.rand(k, n)
        ms = measure_ms(lambda: a @ b, 2, 5)
        gflops = 2.0 * m * k * n / (ms * 1e6)
        print(f"{m:>10}{k:>10}{n:>10}{ms:>15.2f}{gflops:>15.2f}")

    a = torch.rand(4, 256, 256)
    b = torch.rand(4, 256, 256)
    ms = measure_ms(lambda: a @ b, 2, 5)
    print(f"batched 4x(256x256 @ 256x256): {ms:.2f} ms "
          f"({2.0 * 4 * 256 ** 3 / (ms * 1e6):.2f} GFLOPS)")


def bench_elementwise():
    print("\n=== PyTorch element-wise (float32, CPU) ===")
    print(f"{'Op':>10}{'Size':>12}{'Time (ms)':>15}")
    ops = {
        "add": torch.add, "sub": torch.sub, "mul": torch.mul, "div": torch.div,
    }
    for size in [10_000, 100_000, 1_000_000]:
        a = torch.rand(size)
        b = torch.rand(size) + 0.5
        for name, op in ops.items():
            ms = measure_ms(lambda: op(a, b), 3, 100)
            print(f"{name:>10}{size:>12}{ms:>15.4f}")


def bench_vgg16():
    print("\n=== PyTorch VGG16 forward (float32, CPU, eval) ===")
    try:
        import torchvision
    except ImportError:
        print("torchvision not installed; skipping VGG16 reference.")
        return
    model = torchvision.models.vgg16(weights=None).eval()
    print(f"{'Batch':>10}{'Time (ms)':>15}{'img/s':>12}")
    for batch in [1, 2, 4]:
        x = torch.rand(batch, 3, 224, 224)
        ms = measure_ms(lambda: model(x), 1, 3)
        print(f"{batch:>10}{ms:>15.2f}{batch * 1000.0 / ms:>12.2f}")


SECTIONS = {"matmul": bench_matmul, "elementwise": bench_elementwise, "vgg": bench_vgg16}

if __name__ == "__main__":
    print(f"torch {torch.__version__}, threads={torch.get_num_threads()}")
    wanted = sys.argv[1:] or list(SECTIONS)
    for name in wanted:
        if name not in SECTIONS:
            sys.exit(f"unknown section '{name}'; choose from {', '.join(SECTIONS)}")
        SECTIONS[name]()
