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

}
