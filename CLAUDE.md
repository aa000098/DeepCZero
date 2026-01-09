# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepCZero is a deep learning framework implemented from scratch in C++17, inspired by Python's DeZero. It provides automatic differentiation, tensor operations, neural network layers, and training utilities.

## Build Commands

```bash
# Build the shared library (libdeepczero.so)
make

# Run all tests
make test

# Run memory leak tests with valgrind
make memory

# Clean build artifacts
make clean
```

To run a single test, build and run the specific binary:
```bash
make bin/test/container/variable_test
./bin/test/container/variable_test
```

## Dependencies

Required system packages:
- `valgrind` - memory leak checking
- `graphviz` - computation graph visualization
- `libcurl4-gnutls-dev` - data downloading
- `libzip-dev` - model loading

Third-party (in `third_party/`):
- `cnpy` - NumPy file format support for loading pretrained weights

## Architecture

### Core Components

**Tensor** (`include/container/tensor/`)
- `TensorBase<T>` - abstract base class defining tensor interface
- `Tensor1D<T>` - 1D tensor implementation
- `TensorND<T>` - N-dimensional tensor implementation
- `Tensor<T>` - unified wrapper using `shared_ptr<TensorBase<T>>` for polymorphism
- Supports slicing, reshaping, transpose, broadcasting, GEMM operations

**Variable** (`include/container/variable.hpp`)
- Wraps `Tensor<>` with gradient tracking via `VariableImpl<>`
- Stores `creator` pointer to the `Function` that produced it for backpropagation
- `backward()` triggers automatic differentiation through the computation graph

**Function** (`include/function/`)
- Base class using `enable_shared_from_this` for graph construction
- `forward()` computes output, `backward()` computes gradients
- Implementations in subdirectories: `ops/`, activation, loss, conv, pooling, shape functions
- Operator overloads (`+`, `-`, `*`, `/`, `^`) defined in `include/container/variable_ops.hpp`

**Graph** (`include/graph/graph.hpp`)
- Builds computation graph from output `Function*`
- Topological sort for correct backward pass ordering

**Layer** (`include/container/layer/layer.hpp`)
- `Layer` base class manages `params` map and `sublayers` map
- `Linear`, `Conv2d` implementations
- Supports weight save/load and npz format loading

**Model** (`include/container/layer/model.hpp`)
- Inherits from `Layer` for composing networks
- `MLP`, `VGG16` implementations available

**Optimizer** (`include/optimizer/optimizer.hpp`)
- `SGD`, `MomentumSGD` implementations
- Works with `Model` and its `Parameter` instances

**Dataset** (`include/dataset/`)
- `Dataset` base class with transform support
- `SpiralDataset`, `MNISTDataset`, `BigDataset` implementations
- `DataLoader` for batching

### Directory Structure

- `src/` - Implementation files mirroring include structure
- `include/` - Headers; `deepczero.hpp` is the main include
- `test/` - Test files matching src structure; each compiles to separate binary
- `build/` - Object files (generated)
- `bin/` - Test executables and `libdeepczero.so` (generated)

### Testing Pattern

Tests are standalone executables using simple print-based verification:
```cpp
#include "deepczero.hpp"

void test_something() {
    // Test code with manual assertions via printing
}

int main() {
    test_something();
    return 0;
}
```

## Key Implementation Notes

- Uses `shared_ptr`/`weak_ptr` to avoid cyclic references in computation graph
- OpenMP enabled for parallelization (`-fopenmp` flag)
- All tests run in sequence during `make test`; any failure stops the process
