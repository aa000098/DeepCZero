#pragma once

#include "container/tensor/tensor.hpp"

#include <set>

namespace tensor {

template<typename T>
Tensor<T> Tensor<T>::reshape_like(const Tensor<T>& other) const {
	const auto& target_shape = other.get_shape();
	size_t target_size = other.size();

	this->show();
	other.show();
	if (target_size != this->size()) {
		throw std::runtime_error("reshape_like failed: element count mismatch");
	}

	const auto& source_shape = this->get_shape();
	

	return Tensor<T>(target_shape, this->raw_data()[0]);
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

template<typename T>
Tensor<T> Tensor<T>::reshape(const std::vector<size_t>& new_shape) const {
	size_t old_size = impl->size();
	size_t new_size = 1;
	for (size_t d : new_shape) new_size *= d;

	if (old_size != new_size) 
		throw std::runtime_error("reshape error: size mismatch");

	auto data_ptr = impl->shared_data();
	size_t offset = impl->get_offset();

	auto new_strides = compute_contiguous_strides(new_shape);

	auto view_impl = std::make_shared<TensorView<T>>(new_shape, data_ptr, new_strides, offset);

	Tensor<T> result;
	result.impl = view_impl;
	
	return result;
}

template<typename T>
Tensor<T> Tensor<T>::transpose(const std::vector<size_t>& axes) const {
    std::vector<size_t> old_shape = impl->get_shape();
    std::vector<size_t> old_strides = impl->get_strides();

    std::vector<size_t> actual_axes = axes;
    if (actual_axes.empty()) {
		size_t ndim = old_shape.size();
		for (size_t i = 0; i < ndim; i++)
			actual_axes.push_back(ndim-1-i);
    }

    if (actual_axes.size() != old_shape.size())
        throw std::runtime_error("transpose: axes size mismatch");

    std::vector<size_t> new_shape(actual_axes.size());
    std::vector<size_t> new_strides(actual_axes.size());

    for (size_t i = 0; i < actual_axes.size(); ++i) {
        new_shape[i] = old_shape[actual_axes[i]];
        new_strides[i] = old_strides[actual_axes[i]];
    }

    auto view_impl = std::make_shared<TensorView<T>>(
        new_shape,
        impl->shared_data(),
        new_strides,
        impl->get_offset()
    );

    Tensor<T> result;
    result.impl = view_impl;
    return result;
}

template<typename T>
Tensor<T> Tensor<T>::sum(const std::vector<int>& axis, bool keepdims) const {

    const auto& src_shape = impl->get_shape();
    const auto& src_strides = impl->get_strides();
    const auto& src_data = impl->raw_data();

    std::set<size_t> reduce_axis;

	for (int a : axis) {
		if (a < 0) 
			reduce_axis.insert(static_cast<size_t>(a + static_cast<int>(src_shape.size())));
		else
			reduce_axis.insert(static_cast<size_t>(a));
	}

	if (reduce_axis.empty()) {
    	for (size_t i = 0; i < src_shape.size(); ++i)
        	reduce_axis.insert(i);
	}

    // 1. 결과 shape 계산
    std::vector<size_t> result_shape;
    for (size_t i = 0; i < src_shape.size(); ++i) {
        if (reduce_axis.count(i)) {
            if (keepdims) result_shape.push_back(1);
        } else {
            result_shape.push_back(src_shape[i]);
        }
    }

    Tensor<T> result(result_shape, T{});
    auto& result_data = result.raw_data();
    auto result_strides = compute_contiguous_strides(result_shape);

    // 2. 모든 인덱스를 순회하여 값을 더함
    size_t total = src_data.size();
    size_t ndim = src_shape.size();

    for (size_t flat_idx = 0; flat_idx < total; ++flat_idx) {
		// 다차원 인덱스 계산
        std::vector<size_t> idx(ndim);
        size_t remaining = flat_idx;
        for (size_t i = ndim; i-- > 0;) {
            idx[i] = remaining % src_shape[i];
            remaining /= src_shape[i];
        }

        // 결과 텐서 인덱스 계산 (축 제외)
        std::vector<size_t> dst_idx;
        for (size_t i = 0; i < ndim; ++i) {
            if (!reduce_axis.count(i)) {
                dst_idx.push_back(idx[i]);
            } else if (keepdims) {
                dst_idx.push_back(0);
            }
        }

        // flatten index 계산
        size_t dst_flat = 0;
        for (size_t i = 0; i < dst_idx.size(); ++i)
            dst_flat += dst_idx[i] * result_strides[i];

        result_data[dst_flat] += src_data[flat_idx];
    }

    return result;

}

}
