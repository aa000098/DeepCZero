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
Tensor<T> Tensor<T>::reshape(const std::vector<size_t> new_shape) {
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

}
