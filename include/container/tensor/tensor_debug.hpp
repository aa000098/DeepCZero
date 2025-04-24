#pragma once

#include "container/tensor/tensorND.hpp"
#include "container/tensor/tensorview.hpp"

#include <iostream>
#include <string>

namespace tensor {

	template<typename T>
	void print_tensor(
			const TensorND<T>& tensor, 
			size_t depth = 0, 
			size_t offset = 0) {
		const std::vector<size_t>& shape = tensor.get_shape();
		const std::vector<size_t>& strides = tensor.get_strides();
		size_t ndim = tensor.ndim();
		std::string indent(depth * 2, ' ');

		if (depth == ndim - 1) {
			std::cout << indent << "[ ";
        	for (size_t i = 0; i < shape[depth]; ++i) {
				size_t idx = offset + i * strides[depth];
            	std::cout << tensor.raw_data()[idx];
            	if (i != shape[depth] - 1) std::cout << ", ";
        	}
        	std::cout << " ]";
    	} else {
        	std::cout << indent << "[\n";
			size_t dim = shape[depth];
        	for (size_t i = 0; i < dim; ++i) {
            	print_tensor(tensor, depth + 1, offset+i*strides[depth]);
            	if (i != dim - 1) std::cout << "," << std::endl;
        	}
        	std::cout << std::endl << indent << "]";
    	}
	}


	template<typename T>
	void print_tensor(
			const TensorView<T>& tensor, 
			size_t depth = 0, 
			size_t offset = 0) {
		const std::vector<size_t>& shape = tensor.get_shape();
		const std::vector<size_t>& strides = tensor.get_strides();
		size_t ndim = tensor.ndim();
		std::string indent(depth * 2, ' ');

		if (depth == ndim - 1) {
			std::cout << indent << "[ ";
        	for (size_t i = 0; i < shape[depth]; ++i) {
				size_t idx = offset + i * strides[depth];
            	std::cout << tensor.raw_data()[idx];
            	if (i != shape[depth] - 1) std::cout << ", ";
        	}
        	std::cout << " ]";
    	} else {
        	std::cout << indent << "[\n";
			size_t dim = shape[depth];
        	for (size_t i = 0; i < dim; ++i) {
            	print_tensor(tensor, depth + 1, offset+i*strides[depth]);
            	if (i != dim - 1) std::cout << "," << std::endl;
        	}
        	std::cout << std::endl << indent << "]";
    	}
	}

}
