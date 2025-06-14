#pragma once

#include "container/tensor/tensorview.hpp"

#include <cmath>
#include <stdexcept>
#include <iostream>

namespace tensor {

	template<typename T>
	TensorView<T>::TensorView(	
			const std::vector<size_t>& shape, 
			const std::vector<T>& init_data)
		: shape(shape) {
		if (shape.size() < 2)
			throw std::runtime_error("Tensor must be at least 2D to define View tensor");

		size_t expected_size = 1;
		for (auto s : shape) expected_size *= s;

		if (expected_size != init_data.size())
			throw std::invalid_argument("TensorView: shape does not match size of init_data:");

		data_ptr = std::make_shared<std::vector<T>>(init_data);

		compute_strides(); 
	};


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


	template<typename T>
	std::shared_ptr<TensorBase<T>> TensorView<T>::slice(size_t dim, size_t start, size_t end) const {
		if (dim >= shape.size() || start >= end || end > shape[dim])
			throw std::invalid_argument("Invalid slice range");

		std::vector<size_t> new_shape = shape;
		new_shape[dim] = end - start;

		std::vector<size_t> new_strides = strides;
		size_t new_offset = offset + start * strides[dim];

		return std::make_shared<TensorView<T>>(new_shape, this->data_ptr, new_strides, new_offset);
	} 

	template<typename T>
	std::shared_ptr<TensorBase<T>> TensorView<T>::gather_rows(const std::vector<size_t> &indices) const {

	    const std::vector<size_t>& shape = this->get_shape();
    	size_t ndim = shape.size();

    	if (ndim < 1)
        	throw std::runtime_error("get_item: scalar Variable cannot be indexed.");

    	// 슬라이싱 대상: 첫 번째 차원
    	size_t subblock_size = 1;
    	for (size_t i = 1; i < ndim; ++i)
        	subblock_size *= shape[i];

    	std::vector<size_t> new_shape = shape;
    	new_shape[0] = indices.size();
    	std::vector<T> new_data;
    	new_data.reserve(indices.size() * subblock_size);

    	for (size_t i : indices) {
        	if (i >= shape[0])
            	throw std::out_of_range("get_item: index out of bounds");

			size_t stride_0 = this->get_strides()[0];
			size_t base_offset = offset + i * stride_0;

        	for (size_t j = 0; j < subblock_size; ++j) {
            	// 전체 플랫 인덱스 계산: row offset + local offset
				std::vector<size_t> rel_idx(ndim - 1);
				size_t temp = j;
	            for (int d = static_cast<int>(ndim) - 2; d >= 0; --d) {
    	            rel_idx[d] = temp % shape[d + 1];
        	        temp /= shape[d + 1];
            	}

	            size_t actual_offset = base_offset;
    	        for (size_t k = 0; k < rel_idx.size(); ++k)
        	        actual_offset += rel_idx[k] * strides[k + 1];
            	new_data.push_back(this->raw_data()[actual_offset]);
        	}
    	}

    	// 새 shape: 첫 번째 차원만 변경, 나머지는 유지

    	return std::make_shared<TensorView<T>>(TensorView<T>(new_shape, new_data));
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

	template<typename T>
	void TensorView<T>::compute_strides() {
		strides.resize(shape.size());
		size_t stride = 1;
		for (int i = shape.size() - 1; i >= 0; --i) {
			strides[i] = stride;
			stride *= shape[i];
		}
	}
}
