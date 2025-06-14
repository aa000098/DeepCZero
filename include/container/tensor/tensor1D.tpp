#pragma once

#include "container/tensor/tensorbase.hpp"
#include "container/tensor/tensor1D.hpp"

namespace tensor {
	
template <typename T>
std::shared_ptr<TensorBase<T>> Tensor1D<T>::slice(size_t dim, size_t start, size_t end) const {
	if (dim != 0)
		throw std::invalid_argument("Tensor1D only supports slicing along dimension 0.");
	if (start >= end || end > data_ptr->size())
		throw std::out_of_range("Invalid slice range for Tensor1D");

	std::vector<T> sliced(data_ptr->begin() + start, data_ptr->begin() + end);
	return std::make_shared<Tensor1D<T>>(sliced);
}

template <typename T>
std::shared_ptr<TensorBase<T>> Tensor1D<T>::gather_rows(const std::vector<size_t>& indices) const {
	std::vector<T> new_data;
	for (size_t idx : indices) {
		if (idx >= this->size())
			throw std::out_of_range("Index out of range in Tensor1D gather_rows");
		new_data.push_back(this->raw_data()[idx]);
	}
	return std::make_shared<Tensor1D<T>>(new_data);
}

}
