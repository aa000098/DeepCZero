#pragma once

#include "container/tensor/tensor_all.hpp" 

#include <vector>
#include <cstddef>

class Variable;

Variable pooling(const Variable& x,
				std::pair<size_t, size_t> kernel_size,
				std::pair<size_t, size_t> stride = {1,1},
				std::pair<size_t, size_t> pad = {0,0});

Variable pooling(const Variable& x,
				std::initializer_list<size_t> kernel_size,
				std::initializer_list<size_t> stride = {1,1},
				std::initializer_list<size_t> pad = {0,0});

Variable pooling2d_grad(const Variable& gy,
					tensor::Tensor<size_t>& indexes,
					std::vector<size_t> input_shape,
					std::pair<size_t, size_t> kernel_size,
					std::pair<size_t, size_t> stride,
					std::pair<size_t, size_t> pad);

Variable pooling2d_with_indexes(
		const Variable& x,
		tensor::Tensor<size_t>& indexes,
		std::vector<size_t> input_shape,
		std::pair<size_t, size_t> kernel_size,
		std::pair<size_t, size_t> stride,
		std::pair<size_t, size_t> pad);
