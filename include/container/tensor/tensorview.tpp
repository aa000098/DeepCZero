#pragma once

#include "container/tensor/tensorview.hpp"

#include <cmath>
#include <stdexcept>
#include <iostream>

namespace tensor {
	template<typename T>
	TensorView<T> TensorView<T>::operator[](size_t idx) const {
		if (shape.empty())
			throw std::out_of_range("TensorView index out of bounds");
		if (idx >= shape[0])
			throw std::out_of_range("TensorView index out of bounds");
	
		std::vector<size_t> new_shape(shape.begin() + 1, shape.end());
		std::vector<size_t> new_strides(strides.begin() + 1, strides.end());
		size_t new_offset = offset + idx * strides[0];

		return TensorView<T>(new_shape, data_ptr, new_strides, new_offset);
	}

	template<typename T>
	T& TensorView<T>::operator()(const std::vector<size_t>& indices) {
		if (indices.size() != shape.size())
			throw std::invalid_argument("Incorrect number of indices");

		size_t flat_index = offset;
		for (size_t i = 0; i < indices.size(); ++i)
			flat_index += indices[i] * strides[i];

		return (*data_ptr)[flat_index];
	}

	template<typename T>
	size_t TensorView<T>::size() const { 
		size_t total = 1;
		for (auto s : shape) total *= s;
		return total;
	}


	/*
	template<typename T>
	std::ostream& operator<<(std::ostream& os, const TensorView<T>& view) {
		os << "[";
		for (size_t i = 0; i < view.size(); i++) {
			os << view({i});
			if (i != view.size() - 1) os << ", ";
		}
		os << "]";
		return os;
	}
	*/
	template<typename T>
	void TensorView<T>::show() const {
		std::cout << "Data: \n";	
		print_tensor(*this, 0, offset);
		std::cout << std::endl;
	}
}
