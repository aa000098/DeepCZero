#pragma once

#include "container/tensor/tensorview.hpp"
#include "container/tensor/tensorND.hpp"

#include <stdexcept>
#include <iostream>

namespace tensor {
	template<typename T>
	TensorND<T>::TensorND(	
			const std::vector<size_t>& shape, 
			T init) : shape(shape) {
		if (shape.size() < 2)
			throw std::runtime_error("Tensor must be at least 2D to define ND tensor");

		size_t total_size = 1;
		for (auto dim : shape) total_size *= dim;
		data = std::vector<T>(total_size, init);

		compute_strides();
	}

	template<typename T>
	TensorND<T>::TensorND(	
			const std::vector<size_t>& shape, 
			const std::vector<T>& init_data)
		: shape(shape), data(init_data) {
		if (shape.size() < 2)
			throw std::runtime_error("Tensor must be at least 2D to define ND tensor");

		size_t expected_size = 1;
		for (auto s : shape) expected_size *= s;

		if (expected_size != init_data.size())
			throw std::invalid_argument("TensorND: shape does not match size of init_data:");

		compute_strides(); 
	};


	template<typename T>
	TensorView<T> TensorND<T>::operator[](size_t idx) {
		if (shape.size() < 2)
			throw std::runtime_error("Tensor must be at least 2D to support [] access returning view");

		std::vector<size_t> new_shape(shape.begin() + 1, shape.end());
		std::vector<size_t> new_strides(strides.begin() + 1, strides.end());
		size_t offset = idx * strides[0];

		return TensorView<T>(new_shape, new_strides, data, offset);
	}

	template<typename T>
	const TensorView<T> TensorND<T>::operator[](size_t idx) const {
		if (shape.size() < 2)
			throw std::runtime_error("Tensor must be at least 2D to support [] access returning view");

		std::vector<size_t> new_shape(shape.begin() + 1, shape.end());
		std::vector<size_t> new_strides(strides.begin() + 1, strides.end());
		size_t offset = idx * strides[0];

		return TensorView<T>(new_shape, new_strides, data, offset);
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

	template<typename T>
	size_t TensorND<T>::flatten_index(const std::vector<size_t>& indices) {
		if (indices.size() != shape.size())
			throw std::invalid_argument("Dimension mismatch in %s", __func__);
		size_t idx = 0;
		for (size_t i = 0; i < indices.size(); ++i) {
			if (indices[i] >= shape[i])
				throw std::out_of_range("Index out of bounds");
			idx += indices[i] * strides[i];
		}
		return idx;
	}

	template<typename T>
	void print_tensor(
			const TensorND<T>& tensor, 
			size_t depth = 0, 
			size_t offset = 0) {
		const std::vector<size_t> shape = tensor.get_shape();
		size_t ndim = tensor.ndim();
		std::string indent(depth * 2, ' ');

		if (depth == ndim-1) {
			std::cout << indent << "[ ";
        	for (size_t i = offset*shape[depth]; i < (offset+1)*shape[depth]; ++i) {
            	std::cout << tensor.raw_data()[i];
            	if (i != (offset+1)*shape[depth] - 1) std::cout << ", ";
        	}
        	std::cout << " ]";
    	} else {
        	std::vector<size_t> shape = tensor.get_shape();
        	size_t dim = shape[depth];
        	std::cout << indent << "[\n";
        	for (size_t i = 0; i < dim; ++i) {
            	print_tensor(tensor, depth + 1, offset+i);
            	if (i != dim - 1) std::cout << "," << std::endl;
        	}
        	std::cout << std::endl << indent << "]";
    	}
	}

	template<typename T>
	void TensorND<T>::show() {
		std::cout << "Data: \n";	
		print_tensor(*this);
		std::cout << std::endl;
	}
}
