#pragma once

#include "container/tensor/tensor.hpp"

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

}
