#pragma once

#include <numeric>
#include <set>
#include <vector>

namespace tensor {

inline size_t product(const std::vector<size_t>& shape) {
    return std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>());
}

inline std::vector<size_t> unflatten_index(size_t flat_index, const std::vector<size_t>& shape) {
    std::vector<size_t> indices(shape.size());
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        indices[i] = flat_index % shape[i];
        flat_index /= shape[i];
    }
    return indices;
}

inline size_t flatten_index(const std::vector<size_t>& indices, const std::vector<size_t>& strides) {
	size_t flat = 0;
	for (size_t i = 0; i < indices.size(); ++i) {
		flat += indices[i] * strides[i];
	}
	return flat;
}

inline std::vector<size_t> batch_matrix_offsets(const std::vector<size_t>& operand_batch_shape,
												const std::vector<size_t>& batch_shape,
												size_t matrix_size) {
	std::vector<size_t> strides(batch_shape.size(), 0);
	size_t stride = matrix_size;
	for (size_t i = batch_shape.size(); i-- > 0;) {
		strides[i] = (operand_batch_shape[i] == 1) ? 0 : stride;
		stride *= operand_batch_shape[i];
	}

	std::vector<size_t> offsets(product(batch_shape));
	for (size_t b = 0; b < offsets.size(); ++b)
		offsets[b] = flatten_index(unflatten_index(b, batch_shape), strides);
	return offsets;
}

inline std::set<size_t> normalize_axes(const std::vector<int>& axes, size_t ndim) {
    std::set<size_t> result;
    for (int a : axes) {
        if (a < 0) result.insert(static_cast<size_t>(ndim + a));
        else result.insert(static_cast<size_t>(a));
    }
    return result;
}

inline std::vector<size_t> compute_reduced_shape(
    const std::vector<size_t>& shape, 
    const std::set<size_t>& reduce_axes, 
    bool keepdims) 
{
    std::vector<size_t> result;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (reduce_axes.count(i)) {
            if (keepdims) result.push_back(1);
        } else {
            result.push_back(shape[i]);
        }
    }
    return result;
}


inline std::vector<size_t> compute_contiguous_strides(const std::vector<size_t>& shape) {
    std::vector<size_t> strides(shape.size());
    if (!shape.empty()) {
        strides.back() = 1;
        for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }
    return strides;
}

inline std::vector<size_t> replace_axis(const std::vector<size_t>& idx, size_t axis, size_t value) {
	std::vector<size_t> new_idx = idx;
	new_idx[axis] = value;
	return new_idx;
}

template <typename T = float>
inline std::vector<T> arrange_vector(size_t size) {
	std::vector<T> values(size);
	std::iota(values.begin(), values.end(), 0);

	return values;
}

inline std::vector<size_t> cast_axes(const std::vector<int>& axes, size_t ndim) {
	std::vector<size_t> result;
	for (int ax : axes) {
		if (ax < 0)
			result.push_back(static_cast<size_t>(ndim + ax));
		else
			result.push_back(static_cast<size_t>(ax));
	}
	return result;
}

inline size_t get_conv_outsize(int input_size, int kernel_size, int stride, int pad) {
	return (input_size + pad * 2 - kernel_size) / stride + 1;
}

inline size_t get_deconv_outsize(int input_size, int kernel_size, int stride, int pad) {
	return stride * (input_size - 1) + kernel_size - 2*pad;
}

constexpr size_t MAX_STRIDED_GEMM_CALLS = 1024;

struct GemmPlan {
	bool trans_a = false;
	bool trans_b = false;
	size_t M = 0, N = 0, K = 0;
	size_t lda = 0, ldb = 0, ldc = 0;
	std::vector<size_t> a_offsets, b_offsets, c_offsets;
	std::vector<unsigned char> accumulate;
	std::vector<size_t> out_shape;
};

struct GemmOperandLayout {
	bool contract_inner = false;
	std::vector<size_t> k_axes, m_axes;
	std::vector<size_t> outer_free, outer_contract;
	size_t k_size = 1, m_size = 1;
};

inline bool split_gemm_operand(const std::vector<size_t>& shape,
							   const std::vector<size_t>& contract_axes,
							   GemmOperandLayout& layout) {
	size_t ndim = shape.size();
	if (ndim == 0 || contract_axes.empty())
		return false;

	std::vector<unsigned char> is_contract(ndim, 0);
	for (size_t ax : contract_axes) is_contract[ax] = 1;

	unsigned char inner_kind = is_contract[ndim - 1];
	size_t inner_begin = ndim;
	while (inner_begin > 0 && is_contract[inner_begin - 1] == inner_kind) --inner_begin;
	size_t matrix_begin = inner_begin;
	while (matrix_begin > 0 && is_contract[matrix_begin - 1] != inner_kind) --matrix_begin;

	std::vector<size_t> inner_run, prev_run;
	for (size_t ax = inner_begin; ax < ndim; ++ax) inner_run.push_back(ax);
	for (size_t ax = matrix_begin; ax < inner_begin; ++ax) prev_run.push_back(ax);

	layout.contract_inner = inner_kind;
	layout.k_axes = inner_kind ? inner_run : prev_run;
	layout.m_axes = inner_kind ? prev_run : inner_run;
	if (layout.k_axes.empty())
		return false;

	for (size_t ax = 0; ax < matrix_begin; ++ax)
		(is_contract[ax] ? layout.outer_contract : layout.outer_free).push_back(ax);
	for (size_t ax : layout.k_axes) layout.k_size *= shape[ax];
	for (size_t ax : layout.m_axes) layout.m_size *= shape[ax];
	return true;
}

inline bool make_tensordot_plan(const std::vector<size_t>& a_shape,
								const std::vector<size_t>& b_shape,
								const std::vector<size_t>& a_contract,
								const std::vector<size_t>& b_contract,
								GemmPlan& plan) {
	GemmOperandLayout a, b;
	if (!split_gemm_operand(a_shape, a_contract, a) || !split_gemm_operand(b_shape, b_contract, b))
		return false;

	std::vector<size_t> paired(a_shape.size());
	for (size_t i = 0; i < a_contract.size(); ++i) paired[a_contract[i]] = b_contract[i];
	if (a.k_axes.size() != b.k_axes.size())
		return false;
	for (size_t i = 0; i < a.k_axes.size(); ++i)
		if (paired[a.k_axes[i]] != b.k_axes[i])
			return false;

	std::vector<size_t> a_outer_shape, b_outer_shape, k_outer_shape;
	for (size_t ax : a.outer_free) a_outer_shape.push_back(a_shape[ax]);
	for (size_t ax : b.outer_free) b_outer_shape.push_back(b_shape[ax]);
	for (size_t ax : a.outer_contract) k_outer_shape.push_back(a_shape[ax]);
	size_t a_count = product(a_outer_shape);
	size_t b_count = product(b_outer_shape);
	size_t k_count = product(k_outer_shape);
	if (a_count * b_count * k_count > MAX_STRIDED_GEMM_CALLS)
		return false;

	plan.out_shape = a_outer_shape;
	for (size_t ax : a.m_axes) plan.out_shape.push_back(a_shape[ax]);
	plan.out_shape.insert(plan.out_shape.end(), b_outer_shape.begin(), b_outer_shape.end());
	for (size_t ax : b.m_axes) plan.out_shape.push_back(b_shape[ax]);
	if (plan.out_shape.empty())
		return false;

	plan.M = a.m_size;
	plan.N = b.m_size;
	plan.K = a.k_size;
	plan.trans_a = !a.contract_inner;
	plan.lda = a.contract_inner ? a.k_size : a.m_size;
	plan.trans_b = b.contract_inner;
	plan.ldb = b.contract_inner ? b.k_size : b.m_size;
	plan.ldc = b_count * b.m_size;

	auto a_strides = compute_contiguous_strides(a_shape);
	auto b_strides = compute_contiguous_strides(b_shape);
	size_t calls = a_count * b_count * k_count;
	plan.a_offsets.clear(); plan.a_offsets.reserve(calls);
	plan.b_offsets.clear(); plan.b_offsets.reserve(calls);
	plan.c_offsets.clear(); plan.c_offsets.reserve(calls);
	plan.accumulate.clear(); plan.accumulate.reserve(calls);

	for (size_t ai = 0; ai < a_count; ++ai) {
		auto a_idx = unflatten_index(ai, a_outer_shape);
		size_t a_base = 0;
		for (size_t t = 0; t < a_idx.size(); ++t) a_base += a_idx[t] * a_strides[a.outer_free[t]];

		for (size_t bi = 0; bi < b_count; ++bi) {
			auto b_idx = unflatten_index(bi, b_outer_shape);
			size_t b_base = 0;
			for (size_t t = 0; t < b_idx.size(); ++t) b_base += b_idx[t] * b_strides[b.outer_free[t]];

			for (size_t ki = 0; ki < k_count; ++ki) {
				auto k_idx = unflatten_index(ki, k_outer_shape);
				size_t a_off = a_base, b_off = b_base;
				for (size_t t = 0; t < k_idx.size(); ++t) {
					a_off += k_idx[t] * a_strides[a.outer_contract[t]];
					b_off += k_idx[t] * b_strides[paired[a.outer_contract[t]]];
				}
				plan.a_offsets.push_back(a_off);
				plan.b_offsets.push_back(b_off);
				plan.c_offsets.push_back(ai * plan.M * plan.ldc + bi * plan.N);
				plan.accumulate.push_back(ki > 0);
			}
		}
	}
	return true;
}
 

}
