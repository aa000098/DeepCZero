#pragma once

#include "container/tensorview.hpp"

#include <cmath>
#include <stdexcept>
#include <ostream>

namespace tensor {
	template<typename T>
	TensorView<T> TensorView<T>::operator[](size_t idx) const
	{
		if (shape.empty())
			throw std::out_of_range("TensorView index out of bounds");
		if (idx >= shape[0])
			throw std::out_of_range("TensorView index out of bounds");
	
		std::vector<size_t> new_shape(shape.begin() + 1, shape.end());
		std::vector<size_t> new_strides(strides.begin() + 1, strides.end());
		size_t new_offset = offset + idx * strides[0];

		return TensorView<T>(data_ptr, new_shape, new_strides, new_offset);
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
		for (auto s : shape) total * s;
		return total;
	}
/*
	template<typename T>
	TensorView<T>& operator+=(TensorView<T>& lhs, const TensorView<T>& rhs) {
		if (lhs.get_shape() != rhs.get_shape())
			throw std::invalid_argument("Shape mismatch in TensorView += operator");
		
		for (size_t i = 0; i < lhs.size(); i++) 
			lhs.raw_data()[i] += rhs({i});

		return lhs;
	}

	template<typename T>
	T operator*(TensorView<T> a, const TensorView<T>& b) {
		if (a.get_shape() != b.get_shape())
			throw std::invalid_argument("Shape mismatch in TensorView += operator");
		T result;
		std::vector<T> a_data = a.raw_data();
		std::vector<T> b_data = b.raw_data();
		for (size_t i = 0; i < a.size(); i++) 
			result += a_data[i] * b_data[i];
		return result;
	}

	template<typename T>
	TensorView<T>& operator+(TensorView<T> a, const TensorView<T>& b) {
		if (a.get_shape() != b.get_shape())
			throw std::invalid_argument("Shape mismatch in TensorView += operator");
		std::vector<T> result_data(a.size());
		for (int i = 0; i < a.size(); i++) 
			result_data[i] = a.raw_data()[i] + b.raw_data()[i];
		auto shape = a.get_shape();
		auto strides = a.get_strides();
		auto data_ptr = std::make_shared<std::vector<T>>(result_data);

		TensorView<T> result(shape, strides, data_ptr);
		return result;	
	}
*/
	template<typename T>
	TensorView<T>& TensorView<T>::exp() {
		for (size_t i = 0; i < this->size(); i++)
			(*data_ptr)[i] = std::exp((*data_ptr)[i]);
		return *this;
	}

	template<typename T>
	TensorView<T>& TensorView<T>::pow(size_t mul) {
		for (size_t i = 0; i < this->size(); i++)
			(*data_ptr)[i] = std::pow((*data_ptr)[i], mul);
		return *this;
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
}
