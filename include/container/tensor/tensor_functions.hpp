#pragma once

#include "container/tensor/tensor.hpp"

namespace tensor {


template<typename T>
Tensor<T> broadcast_to(const Tensor<T>& src, const std::vector<size_t>& target_shape);


template<typename T>
Tensor<T> sum_to(const Tensor<T>& src, const std::vector<size_t>& target_shape);


template<typename T>
void add_at(Tensor<T>& gx, 
			const std::vector<size_t>& slices,
			const Tensor<T>& gy);

template<typename T>
std::tuple<Tensor<T>, Tensor<T>, std::vector<size_t>> broadcast_binary_operands(
    const Tensor<T>& a, const Tensor<T>& b);

}
#include "container/tensor/tensor_functions.tpp"
