#pragma once

#include "container/tensor/tensor.hpp"
#include "container/tensor/tensor_functions.hpp"
#include "container/tensor/tensor_utils.hpp"

#include <omp.h>

#ifdef USE_MKL
#include "container/tensor/tensor_ops_mkl.hpp"
#endif

#include "config/backend_config.hpp"

namespace tensor {

// Naive implementations of element-wise operations

template<typename T>
Tensor<T> add_naive(const Tensor<T>& a, const Tensor<T>& b) {
	auto [a_bc, b_bc, broadcast_shape] = broadcast_binary_operands(a, b);

    Tensor<T> result(broadcast_shape, T{});
    size_t total = result.size();
    size_t ndim_result = broadcast_shape.size();

	//#pragma omp parallel for
    for (size_t flat_idx = 0; flat_idx < total; ++flat_idx) {
        std::vector<size_t> idx(ndim_result);
        size_t rem = flat_idx;
        for (size_t i = ndim_result; i-- > 0;) {
            idx[i] = rem % broadcast_shape[i];
            rem /= broadcast_shape[i];
        }

        result(idx) = a_bc(idx) + b_bc(idx);
    }

	return result;
}

template<typename T>
Tensor<T> sub_naive(const Tensor<T>& a, const Tensor<T>& b) {
	auto [a_bc, b_bc, broadcast_shape] = broadcast_binary_operands(a, b);

    Tensor<T> result(broadcast_shape, T{});
    size_t total = result.size();
    size_t ndim_result = broadcast_shape.size();

	//#pragma omp parallel for
    for (size_t flat_idx = 0; flat_idx < total; ++flat_idx) {
        std::vector<size_t> idx(ndim_result);
        size_t rem = flat_idx;
        for (size_t i = ndim_result; i-- > 0;) {
            idx[i] = rem % broadcast_shape[i];
            rem /= broadcast_shape[i];
        }

        result(idx) = a_bc(idx) - b_bc(idx);
    }

	return result;
}

template<typename T>
Tensor<T> mul_naive(const Tensor<T>& a, const Tensor<T>& b) {
	auto [a_bc, b_bc, broadcast_shape] = broadcast_binary_operands(a, b);

    Tensor<T> result(broadcast_shape, T{});
    size_t total = result.size();
    size_t ndim_result = broadcast_shape.size();

	//#pragma omp parallel for
    for (size_t flat_idx = 0; flat_idx < total; ++flat_idx) {
        std::vector<size_t> idx(ndim_result);
        size_t rem = flat_idx;
        for (size_t i = ndim_result; i-- > 0;) {
            idx[i] = rem % broadcast_shape[i];
            rem /= broadcast_shape[i];
        }

        result(idx) = a_bc(idx) * b_bc(idx);
    }

	return result;
}

template<typename T>
Tensor<T> div_naive(const Tensor<T>& a, const Tensor<T>& b) {
	auto [a_bc, b_bc, broadcast_shape] = broadcast_binary_operands(a, b);

    Tensor<T> result(broadcast_shape, T{});
    size_t total = result.size();
    size_t ndim_result = broadcast_shape.size();

	//#pragma omp parallel for
    for (size_t flat_idx = 0; flat_idx < total; ++flat_idx) {
        std::vector<size_t> idx(ndim_result);
        size_t rem = flat_idx;
        for (size_t i = ndim_result; i-- > 0;) {
            idx[i] = rem % broadcast_shape[i];
            rem /= broadcast_shape[i];
        }

		if (b_bc(idx) == 0)
			throw std::runtime_error("Tensor Div: Division by zero");
        result(idx) = a_bc(idx) / b_bc(idx);
    }

	return result;
}

// Dispatcher functions that use MKL when available and shapes match

template<typename T>
Tensor<T> add(const Tensor<T>& a, const Tensor<T>& b) {
#ifdef USE_MKL
	// Use MKL if shapes match (no broadcasting needed)
	if (a.get_shape() == b.get_shape()) {
		if constexpr (std::is_same<T, float>::value || std::is_same<T, double>::value) {
			return add_mkl(a, b);
		}
	}
#endif
	return add_naive(a, b);
}

template<typename T>
Tensor<T> sub(const Tensor<T>& a, const Tensor<T>& b) {
#ifdef USE_MKL
	if (a.get_shape() == b.get_shape()) {
		if constexpr (std::is_same<T, float>::value || std::is_same<T, double>::value) {
			return sub_mkl(a, b);
		}
	}
#endif
	return sub_naive(a, b);
}

template<typename T>
Tensor<T> mul(const Tensor<T>& a, const Tensor<T>& b) {
#ifdef USE_MKL
	if (a.get_shape() == b.get_shape()) {
		if constexpr (std::is_same<T, float>::value || std::is_same<T, double>::value) {
			return mul_mkl(a, b);
		}
	}
#endif
	return mul_naive(a, b);
}

template<typename T>
Tensor<T> div(const Tensor<T>& a, const Tensor<T>& b) {
#ifdef USE_MKL
	if (a.get_shape() == b.get_shape()) {
		if constexpr (std::is_same<T, float>::value || std::is_same<T, double>::value) {
			return div_mkl(a, b);
		}
	}
#endif
	return div_naive(a, b);
}

template<typename T>
Tensor<T> neg(const Tensor<T>& a) {
	std::vector<T> result_data(a.raw_data());
	for (auto& val : result_data)
		val = -val;

	return Tensor<T>(a.get_shape(), result_data);
}

// inplace
template<typename T>
void add_inplace(Tensor<T>& a, const Tensor<T>& b) {
	const auto& shape_a = a.get_shape();
	Tensor<T> b_bc = (shape_a == b.get_shape()) ? b : broadcast_to(b, shape_a);

	const size_t ndim = shape_a.size();
	const size_t total = a.size();

	//#pragma omp parallel for
	for (size_t flat_idx = 0; flat_idx < total; ++flat_idx) {
		std::vector<size_t> idx(ndim);
		size_t rem = flat_idx;
		for (size_t i = ndim; i-- > 0;) {
			idx[i] = rem % shape_a[i];
			rem /= shape_a[i];
		}
		a(idx) += b_bc(idx);
	}
}

template<typename T>
void sub_inplace(Tensor<T>& a, const Tensor<T>& b) {
	const auto& shape_a = a.get_shape();
	Tensor<T> b_bc = (shape_a == b.get_shape()) ? b : broadcast_to(b, shape_a);

	const size_t ndim = shape_a.size();
	const size_t total = a.size();

	//#pragma omp parallel for
	for (size_t flat_idx = 0; flat_idx < total; ++flat_idx) {
		std::vector<size_t> idx(ndim);
		size_t rem = flat_idx;
		for (size_t i = ndim; i-- > 0;) {
			idx[i] = rem % shape_a[i];
			rem /= shape_a[i];
		}
		a(idx) -= b_bc(idx);
	}
}

template<typename T>
void mul_inplace(Tensor<T>& a, const Tensor<T>& b) {
	const auto& shape_a = a.get_shape();
	Tensor<T> b_bc = (shape_a == b.get_shape()) ? b : broadcast_to(b, shape_a);

	const size_t ndim = shape_a.size();
	const size_t total = a.size();

	//#pragma omp parallel for
	for (size_t flat_idx = 0; flat_idx < total; ++flat_idx) {
		std::vector<size_t> idx(ndim);
		size_t rem = flat_idx;
		for (size_t i = ndim; i-- > 0;) {
			idx[i] = rem % shape_a[i];
			rem /= shape_a[i];
		}
		a(idx) *= b_bc(idx);
	}
}

template<typename T>
void div_inplace(Tensor<T>& a, const Tensor<T>& b) {
	const auto& shape_a = a.get_shape();
	Tensor<T> b_bc = (shape_a == b.get_shape()) ? b : broadcast_to(b, shape_a);

	const size_t ndim = shape_a.size();
	const size_t total = a.size();

	//#pragma omp parallel for
	for (size_t flat_idx = 0; flat_idx < total; ++flat_idx) {
		std::vector<size_t> idx(ndim);
		size_t rem = flat_idx;
		for (size_t i = ndim; i-- > 0;) {
			idx[i] = rem % shape_a[i];
			rem /= shape_a[i];
		}
		if (b_bc(idx) == static_cast<T>(0))
			throw std::runtime_error("div_inplace: division by zero detected");	
		a(idx) += b_bc(idx);
	}
}

// Tensor ⊕ Scalar
template<typename T>
void add_scalar_inplace(Tensor<T>& a, T scalar) {
	auto& data = a.raw_data();
	for (auto& val : data)
		val += scalar;
}

// Tensor<T> -= scalar
template<typename T>
void sub_scalar_inplace(Tensor<T>& a, T scalar) {
	auto& data = a.raw_data();
	for (auto& val : data)
		val -= scalar;
}

template<typename T>
void mul_scalar_inplace(Tensor<T>& a, T scalar) {
	auto& data = a.raw_data();
	for (auto& val : data)
		val *= scalar;
}

template<typename T>
void div_scalar_inplace(Tensor<T>& a, T scalar) {
	auto& data = a.raw_data();
	if (scalar == static_cast<T>(0))
		throw std::runtime_error("div_scalar_inplace: division by zero detected");	
	for (auto& val : data)
		val /= scalar;
}


// Naive implementations of math functions

template<typename T>
Tensor<T> pow_naive(const Tensor<T>& x, const T scalar) {
	std::vector<T> result_data(x.raw_data());
	const std::vector<T>& x_data = x.raw_data();
	for (size_t i = 0; i < result_data.size(); i++)
		result_data[i] = std::pow(x_data[i], scalar);

	return Tensor<T>(x.get_shape(), result_data);
}

template<typename T>
Tensor<T> exp_naive(const Tensor<T>& x) {
	std::vector<T> result_data(x.raw_data());
	const std::vector<T>& x_data = x.raw_data();
	for (size_t i = 0; i < result_data.size(); i++)
		result_data[i] = std::exp(x_data[i]);

	return Tensor<T>(x.get_shape(), result_data);
}

template<typename T>
Tensor<T> log_naive(const Tensor<T>& x) {
    const std::vector<T>& x_data = x.raw_data();
    std::vector<T> result_data(x_data.size());

    for (size_t i = 0; i < x_data.size(); ++i) {
        if (x_data[i] <= static_cast<T>(0))
            throw std::domain_error("log: input must be positive.");
        result_data[i] = std::log(x_data[i]);
    }

    return Tensor<T>(x.get_shape(), result_data);
}

template<typename T>
Tensor<T> sin_naive(const Tensor<T>& x) {
	std::vector<T> result_data(x.raw_data());
	const std::vector<T>& x_data = x.raw_data();
	for (size_t i = 0; i < result_data.size(); i++)
		result_data[i] = std::sin(x_data[i]);

	return Tensor<T>(x.get_shape(), result_data);
}

template<typename T>
Tensor<T> cos_naive(const Tensor<T>& x) {
	std::vector<T> result_data(x.raw_data());
	const std::vector<T>& x_data = x.raw_data();
	for (size_t i = 0; i < result_data.size(); i++)
		result_data[i] = std::cos(x_data[i]);

	return Tensor<T>(x.get_shape(), result_data);
}

template<typename T>
Tensor<T> tanh_naive(const Tensor<T>& x) {
	std::vector<T> result_data(x.raw_data());
	const std::vector<T>& x_data = x.raw_data();
	for (size_t i = 0; i < result_data.size(); i++)
		result_data[i] = std::tanh(x_data[i]);

	return Tensor<T>(x.get_shape(), result_data);
}

// Dispatcher functions that use MKL when available

template<typename T>
Tensor<T> pow(const Tensor<T>& x, const T scalar) {
#ifdef USE_MKL
	if constexpr (std::is_same<T, float>::value || std::is_same<T, double>::value) {
		return pow_mkl(x, scalar);
	}
#endif
	return pow_naive(x, scalar);
}

template<typename T>
Tensor<T> exp(const Tensor<T>& x) {
#ifdef USE_MKL
	if constexpr (std::is_same<T, float>::value || std::is_same<T, double>::value) {
		return exp_mkl(x);
	}
#endif
	return exp_naive(x);
}

template<typename T>
Tensor<T> log(const Tensor<T>& x) {
#ifdef USE_MKL
	if constexpr (std::is_same<T, float>::value || std::is_same<T, double>::value) {
		return log_mkl(x);
	}
#endif
	return log_naive(x);
}

template<typename T>
Tensor<T> sin(const Tensor<T>& x) {
#ifdef USE_MKL
	if constexpr (std::is_same<T, float>::value || std::is_same<T, double>::value) {
		return sin_mkl(x);
	}
#endif
	return sin_naive(x);
}

template<typename T>
Tensor<T> cos(const Tensor<T>& x) {
#ifdef USE_MKL
	if constexpr (std::is_same<T, float>::value || std::is_same<T, double>::value) {
		return cos_mkl(x);
	}
#endif
	return cos_naive(x);
}

template<typename T>
Tensor<T> tanh(const Tensor<T>& x) {
#ifdef USE_MKL
	if constexpr (std::is_same<T, float>::value || std::is_same<T, double>::value) {
		return tanh_mkl(x);
	}
#endif
	return tanh_naive(x);
}

// Naive (non-MKL) matrix multiplication implementation
template<typename T>
Tensor<T> dot_naive(const Tensor<T>& a, const Tensor<T>& b) {
	// Fallback to naive implementation
	std::vector<size_t> a_shape = a.get_shape();
	std::vector<size_t> b_shape = b.get_shape();

	// TODO: check dot condition
	if (a_shape.size() < 2 || b_shape.size() < 2)
		std::runtime_error("dot: tensors must be at least 2D");


    size_t ndim = std::max(a_shape.size(), b_shape.size());
    while (a_shape.size() < ndim) a_shape.insert(a_shape.begin(), static_cast<size_t>(1));
    while (b_shape.size() < ndim) b_shape.insert(b_shape.begin(), static_cast<size_t>(1));

    // 2. check inner dim compatibility
    size_t M = a_shape[ndim - 2];
    size_t K1 = a_shape[ndim - 1];
    size_t K2 = b_shape[ndim - 2];
    size_t N = b_shape[ndim - 1];

    if (K1 != K2)
        throw std::runtime_error("dot: inner dimensions mismatch");

    // 3. determine broadcasted batch shape
    std::vector<size_t> batch_shape;
    for (size_t i = 0; i < ndim - 2; ++i) {
        if (a_shape[i] == b_shape[i]) batch_shape.push_back(a_shape[i]);
        else if (a_shape[i] == 1)     batch_shape.push_back(b_shape[i]);
        else if (b_shape[i] == 1)     batch_shape.push_back(a_shape[i]);
        else throw std::runtime_error("dot: batch dimension mismatch");
    }

    // 4. final result shape = batch_shape + [M, N]
    std::vector<size_t> result_shape = batch_shape;
	std::vector<size_t> a_bc_shape = batch_shape;
	a_bc_shape.push_back(M);
	a_bc_shape.push_back(K1);

	std::vector<size_t> b_bc_shape = batch_shape;
	b_bc_shape.push_back(K2);
	b_bc_shape.push_back(N);

    // 5. broadcast input tensors
	Tensor<T> a_bc = broadcast_to(a, a_bc_shape);
	Tensor<T> b_bc = broadcast_to(b, b_bc_shape);
    result_shape.push_back(M);
    result_shape.push_back(N);

    std::vector<T> result_data(product(result_shape), T{});
	auto result_strides = compute_contiguous_strides(result_shape); 

    // 6. flatten loop over batches
    size_t batch_size = product(batch_shape);

	#pragma omp parallel for
	for (size_t flat = 0; flat < batch_size * M * N; ++flat) {
		size_t batch_idx = flat / (M * N);
		size_t i = (flat / N) % M;
		size_t j = flat % N;

		auto batch_idx_vec = unflatten_index(batch_idx, batch_shape);

		T sum = T{};
		for (size_t k = 0; k < K1; ++k) {
			auto idx_a = batch_idx_vec; idx_a.push_back(i); idx_a.push_back(k);
			auto idx_b = batch_idx_vec; idx_b.push_back(k); idx_b.push_back(j);
			sum += a_bc(idx_a) * b_bc(idx_b);
		}

		auto idx_res = batch_idx_vec; idx_res.push_back(i); idx_res.push_back(j);
		size_t flat_idx = flatten_index(idx_res, result_strides);
		result_data[flat_idx] = sum;
	}
	/*
    for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        auto batch_idx_vec = unflatten_index(batch_idx, batch_shape);

        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                T sum = T{};
                for (size_t k = 0; k < K1; ++k) {
                    auto idx_a = batch_idx_vec; idx_a.push_back(i); idx_a.push_back(k);
                    auto idx_b = batch_idx_vec; idx_b.push_back(k); idx_b.push_back(j);
                    sum += a_bc(idx_a) * b_bc(idx_b);
                }
                auto idx_res = batch_idx_vec; idx_res.push_back(i); idx_res.push_back(j);
                size_t flat = flatten_index(idx_res, result_strides);
                result_data[flat] = sum;
            }
        }
    }
*/
    return Tensor<T>(result_shape, result_data);
}

// Main dot function that dispatches to appropriate implementation
template<typename T>
Tensor<T> dot(const Tensor<T>& a, const Tensor<T>& b) {
#ifdef USE_MKL
	if constexpr (std::is_same<T, float>::value || std::is_same<T, double>::value) {
		if (dcz::BackendConfig::get().use_mkl) {
			return dot_mkl(a, b);
		}
	}
#endif
	return dot_naive(a, b);
}

template<typename T>
Tensor<T> tensordot(const Tensor<T> &A, 
                    const Tensor<T>& B,
                    const std::pair<std::vector<int>, 
							std::vector<int>>& axes) {
    // 1. 분해
    const auto A_shape = A.get_shape();
    const auto B_shape = B.get_shape();
    const size_t A_ndim = A_shape.size();
    const size_t B_ndim = B_shape.size();

    const auto& A_contract_axes = cast_axes(axes.first, A_ndim);
    const auto& B_contract_axes = cast_axes(axes.second, B_ndim);

    if (A_contract_axes.size() != B_contract_axes.size())
        throw std::runtime_error("Axes length mismatch");

    for (size_t i = 0; i < A_contract_axes.size(); ++i) {
        if (A_shape[A_contract_axes[i]] != B_shape[B_contract_axes[i]])
            throw std::runtime_error("Shape mismatch in tensordot axes");
    }

    // 2. 계산에 사용될 축 외의 남은 축 구하기
    std::vector<size_t> A_free_axes, B_free_axes;
    for (size_t i = 0; i < A_ndim; ++i)
        if (std::find(A_contract_axes.begin(), A_contract_axes.end(), i) == A_contract_axes.end())
            A_free_axes.push_back(i);
    for (size_t i = 0; i < B_ndim; ++i)
        if (std::find(B_contract_axes.begin(), B_contract_axes.end(), i) == B_contract_axes.end())
            B_free_axes.push_back(i);

    // 3. transpose
    std::vector<size_t> A_axes = A_free_axes;
    A_axes.insert(A_axes.end(), A_contract_axes.begin(), A_contract_axes.end());

    std::vector<size_t> B_axes = B_contract_axes;
    B_axes.insert(B_axes.end(), B_free_axes.begin(), B_free_axes.end());

    Tensor<T> A_trans = A.transpose(A_axes).contiguous();
    Tensor<T> B_trans = B.transpose(B_axes).contiguous();

    // 4. reshape
    size_t A_outer = 1, A_inner = 1;
    for (size_t ax : A_free_axes) A_outer *= A_shape[ax];
    for (size_t ax : A_contract_axes) A_inner *= A_shape[ax];

    size_t B_inner = 1, B_outer = 1;
    for (size_t ax : B_contract_axes) B_inner *= B_shape[ax];
    for (size_t ax : B_free_axes) B_outer *= B_shape[ax];

    Tensor<T> A_mat = A_trans.reshape({A_outer, A_inner});
    Tensor<T> B_mat = B_trans.reshape({B_inner, B_outer});

    // 5. 행렬 곱 (A_inner == B_inner)
    Tensor<T> result = dot(A_mat, B_mat); // (A_outer, B_outer)

    // 6. reshape to final shape
    std::vector<size_t> A_free_shape, B_free_shape;
    for (size_t ax : A_free_axes) A_free_shape.push_back(A_shape[ax]);
    for (size_t ax : B_free_axes) B_free_shape.push_back(B_shape[ax]);

    std::vector<size_t> final_shape = A_free_shape;
    final_shape.insert(final_shape.end(), B_free_shape.begin(), B_free_shape.end());

    return result.reshape(final_shape);	
}

template<typename T>
Tensor<T> maximum(const Tensor<T>& input, T scalar) {
	std::vector<T> out_data(input.size());
	const auto& in_data = input.data();

	for (size_t i = 0; i < input.size(); i++) 
		out_data[i] = std::max(in_data[i], scalar);

	return Tensor<T>(input.get_shape(), out_data);
}

template<typename T>
Tensor<T> greater(const Tensor<T>& x, T scalar) {
	const auto& x_data = x.data();
	std::vector<T> out_data(x.size());

	for (size_t i = 0; i < x.size(); i++)
		out_data[i] = x_data[i] > scalar ? 1 : 0;

	return Tensor<T>(x.get_shape(), out_data);

}

}
