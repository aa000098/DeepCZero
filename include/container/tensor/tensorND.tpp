#pragma once

#include "container/tensor/tensorview.hpp"
#include "container/tensor/tensorND.hpp"
#include "container/tensor/tensor_debug.hpp"
#include "container/tensor/tensor_utils.hpp"

#include <stdexcept>
#include <iostream>
#include <memory>

namespace tensor {
	template<typename T>
	TensorND<T>::TensorND(	
			const std::vector<size_t>& shape, 
			T init) : shape(shape) {
		if (shape.size() < 2)
			throw std::runtime_error("Tensor must be at least 2D to define ND tensor");

		size_t total_size = 1;
		for (auto dim : shape) total_size *= dim;
		data_ptr = std::make_shared<std::vector<T>>(total_size, init);

		compute_strides();
	}

	template<typename T>
	TensorND<T>::TensorND(	
			const std::vector<size_t>& shape, 
			const std::vector<T>& init_data)
		: shape(shape) {
		if (shape.size() < 2)
			throw std::runtime_error("Tensor must be at least 2D to define ND tensor");

		size_t expected_size = 1;
		for (auto s : shape) expected_size *= s;

		if (expected_size != init_data.size())
			throw std::invalid_argument("TensorND: shape does not match size of init_data:");

		data_ptr = std::make_shared<std::vector<T>>(init_data);

		compute_strides(); 
	};

	template<typename T>
	std::shared_ptr<TensorBase<T>> TensorND<T>::slice(size_t dim, size_t start, size_t end) const {
		if (dim >= shape.size() || start >= end || end > shape[dim])
			throw std::invalid_argument("Invalid slice range");

		std::vector<size_t> new_shape = shape;
		new_shape[dim] = end - start;

		TensorND<T> result(new_shape);

		std::vector<size_t> idx(shape.size(), 0);
		for (size_t flat_idx = 0; flat_idx < result.size(); flat_idx++) {
			auto new_idx = unflatten_index(flat_idx, new_shape);
			idx = new_idx;
			idx[dim] += start;
			result(new_idx) = (*this)(idx);
		}

		return std::make_shared<TensorND<T>>(result);
	}

	template<typename T>
	std::shared_ptr<TensorBase<T>> TensorND<T>::gather_rows(const std::vector<size_t> &indices) const {

	    const std::vector<size_t>& shape = this->get_shape();
    	size_t ndim = shape.size();

    	if (ndim == 0)
        	throw std::runtime_error("get_item: scalar Variable cannot be indexed.");

    	// 슬라이싱 대상: 첫 번째 차원
    	size_t subblock_size = 1;
    	for (size_t i = 1; i < ndim; ++i)
        	subblock_size *= shape[i];

    	std::vector<T> new_data;
    	new_data.reserve(indices.size() * subblock_size);

    	for (size_t i : indices) {
        	if (i >= shape[0])
            throw std::out_of_range("get_item: index out of bounds");

        	for (size_t j = 0; j < subblock_size; ++j) {
            	// 전체 플랫 인덱스 계산: row offset + local offset
            	size_t flat_idx = i * subblock_size + j;
            	new_data.push_back(this->raw_data()[flat_idx]);
        	}
    	}

    	// 새 shape: 첫 번째 차원만 변경, 나머지는 유지
    	std::vector<size_t> new_shape = shape;
    	new_shape[0] = indices.size();

    	return std::make_shared<TensorND<T>>(TensorND<T>(new_shape, new_data));
	}


	template<typename T>
	std::shared_ptr<TensorBase<T>> TensorND<T>::operator[](size_t idx) {
		if (shape.size() < 2)
			throw std::runtime_error("Tensor must be at least 2D to support [] access returning view");

		std::vector<size_t> new_shape(shape.begin() + 1, shape.end());
		std::vector<size_t> new_strides(strides.begin() + 1, strides.end());
		size_t new_offset = idx * strides[0];

		return std::make_shared<TensorView<T>>(new_shape, data_ptr, new_strides, new_offset);
	}

	template<typename T>
	const std::shared_ptr<TensorBase<T>> TensorND<T>::operator[](size_t idx) const {
		if (shape.size() < 2)
			throw std::runtime_error("Tensor must be at least 2D to support [] access returning view");

		std::vector<size_t> new_shape(shape.begin() + 1, shape.end());
		std::vector<size_t> new_strides(strides.begin() + 1, strides.end());
		size_t new_offset = idx * strides[0];

		return std::make_shared<TensorView<T>>(new_shape, data_ptr, new_strides, new_offset);
	}

	template<typename T>
	TensorView<T> TensorND<T>::view(size_t index) const {
		if (shape.size() < 2)
			throw std::runtime_error("Cannot slice 1D tensor");

		std::vector<size_t> new_shape(shape.begin() + 1, shape.end());
		std::vector<size_t> new_strides(strides.begin() + 1, strides.end());
		size_t new_offset = index * strides[0];

		return TensorView<T>(new_shape, data_ptr, new_strides, new_offset);
	}

	template<typename T>
	void TensorND<T>::compute_strides() {
		strides.resize(shape.size());
		size_t stride = 1;
		for (int i = shape.size() - 1; i >= 0; --i) {
			strides[i] = stride;
			stride *= shape[i];
		}
	}
/*
	template<typename T>
	size_t TensorND<T>::flatten_index(const std::vector<size_t>& indices) {
		if (indices.size() != shape.size())
			throw std::invalid_argument("Dimension mismatch in " + std::string(__func__));
		size_t idx = 0;
		for (size_t i = 0; i < indices.size(); ++i) {
			if (indices[i] >= shape[i])
				throw std::out_of_range("Index out of bounds");
			idx += indices[i] * strides[i];
		}
		return idx;
	}
*/
	template<typename T>
	void TensorND<T>::show() const {
		print_tensor(*this);
		std::cout << std::endl;
	}
}
