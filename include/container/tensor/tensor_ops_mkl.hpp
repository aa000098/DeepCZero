#pragma once

#ifdef USE_MKL

#include <mkl_cblas.h>
#include "container/tensor/tensor.hpp"
#include "container/tensor/tensor_functions.hpp"
#include "container/tensor/tensor_utils.hpp"

namespace tensor {

// Helper to determine if we can use MKL for this operation
template<typename T>
constexpr bool can_use_mkl() {
    return std::is_same<T, float>::value || std::is_same<T, double>::value;
}

// MKL-optimized matrix multiplication for contiguous 2D matrices
template<typename T>
void mkl_gemm_2d(const T* A, const T* B, T* C,
                 size_t M, size_t K, size_t N,
                 bool transA = false, bool transB = false) {
    // MKL uses column-major by default, but we use row-major
    // So we compute: C^T = B^T * A^T which gives us C = A * B in row-major

    const CBLAS_LAYOUT layout = CblasRowMajor;
    const CBLAS_TRANSPOSE trans_a = transA ? CblasTrans : CblasNoTrans;
    const CBLAS_TRANSPOSE trans_b = transB ? CblasTrans : CblasNoTrans;

    const MKL_INT m = static_cast<MKL_INT>(M);
    const MKL_INT k = static_cast<MKL_INT>(K);
    const MKL_INT n = static_cast<MKL_INT>(N);

    const T alpha = static_cast<T>(1.0);
    const T beta = static_cast<T>(0.0);

    // lda, ldb, ldc: leading dimensions (stride between rows in row-major)
    const MKL_INT lda = transA ? m : k;
    const MKL_INT ldb = transB ? k : n;
    const MKL_INT ldc = n;

    if constexpr (std::is_same<T, float>::value) {
        cblas_sgemm(layout, trans_a, trans_b,
                    m, n, k,
                    alpha, A, lda, B, ldb,
                    beta, C, ldc);
    } else if constexpr (std::is_same<T, double>::value) {
        cblas_dgemm(layout, trans_a, trans_b,
                    m, n, k,
                    alpha, A, lda, B, ldb,
                    beta, C, ldc);
    }
}

// MKL-optimized dot product with batching support
template<typename T>
Tensor<T> dot_mkl(const Tensor<T>& a, const Tensor<T>& b) {
    static_assert(can_use_mkl<T>(), "MKL only supports float and double types");

    std::vector<size_t> a_shape = a.get_shape();
    std::vector<size_t> b_shape = b.get_shape();

    if (a_shape.size() < 2 || b_shape.size() < 2)
        throw std::runtime_error("dot_mkl: tensors must be at least 2D");

    // Normalize dimensions
    size_t ndim = std::max(a_shape.size(), b_shape.size());
    while (a_shape.size() < ndim) a_shape.insert(a_shape.begin(), 1);
    while (b_shape.size() < ndim) b_shape.insert(b_shape.begin(), 1);

    // Extract matrix dimensions
    size_t M = a_shape[ndim - 2];
    size_t K1 = a_shape[ndim - 1];
    size_t K2 = b_shape[ndim - 2];
    size_t N = b_shape[ndim - 1];

    if (K1 != K2)
        throw std::runtime_error("dot_mkl: inner dimensions mismatch");

    size_t K = K1;

    // Determine batch shape
    std::vector<size_t> batch_shape;
    for (size_t i = 0; i < ndim - 2; ++i) {
        if (a_shape[i] == b_shape[i]) batch_shape.push_back(a_shape[i]);
        else if (a_shape[i] == 1)     batch_shape.push_back(b_shape[i]);
        else if (b_shape[i] == 1)     batch_shape.push_back(a_shape[i]);
        else throw std::runtime_error("dot_mkl: batch dimension mismatch");
    }

    // Build result shape
    std::vector<size_t> result_shape = batch_shape;
    result_shape.push_back(M);
    result_shape.push_back(N);

    // Broadcast inputs if needed
    std::vector<size_t> a_bc_shape = batch_shape;
    a_bc_shape.push_back(M);
    a_bc_shape.push_back(K);

    std::vector<size_t> b_bc_shape = batch_shape;
    b_bc_shape.push_back(K);
    b_bc_shape.push_back(N);

    Tensor<T> a_bc = broadcast_to(a, a_bc_shape);
    Tensor<T> b_bc = broadcast_to(b, b_bc_shape);

    // Make contiguous for MKL
    a_bc = a_bc.contiguous();
    b_bc = b_bc.contiguous();

    // Allocate result
    std::vector<T> result_data(product(result_shape), T{});

    size_t batch_size = product(batch_shape);
    size_t a_matrix_size = M * K;
    size_t b_matrix_size = K * N;
    size_t c_matrix_size = M * N;

    const T* a_ptr = a_bc.raw_data().data();
    const T* b_ptr = b_bc.raw_data().data();
    T* c_ptr = result_data.data();

    // Process each batch using MKL GEMM
    #pragma omp parallel for
    for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        const T* a_batch = a_ptr + batch_idx * a_matrix_size;
        const T* b_batch = b_ptr + batch_idx * b_matrix_size;
        T* c_batch = c_ptr + batch_idx * c_matrix_size;

        mkl_gemm_2d(a_batch, b_batch, c_batch, M, K, N);
    }

    return Tensor<T>(result_shape, result_data);
}

} // namespace tensor

#endif // USE_MKL
