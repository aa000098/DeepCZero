#pragma once

#include "container/tensor/tensor.hpp"
#include "container/tensor/tensorview.hpp"

#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <numeric>

namespace tensor {


template<typename T>
Tensor<T> broadcast_to(const Tensor<T>& src, const std::vector<size_t>& target_shape) {

	const auto& src_shape = src.get_shape();
	const auto& src_strides = src.get_strides();

	size_t ndim_src = src_shape.size();
	size_t ndim_target = target_shape.size();

	std::vector<size_t> padded_shape = src_shape;
	std::vector<size_t> padded_strides = src_strides;

	if (ndim_src < ndim_target) {
		size_t diff = ndim_target - ndim_src;
		padded_shape.insert(padded_shape.begin(), diff, 1);
		padded_strides.insert(padded_strides.begin(), diff, 0);
	}

	std::vector<size_t> new_strides(ndim_target);
	for (size_t i = 0; i < ndim_target; i++) {
		if (padded_shape[i] == target_shape[i])
			new_strides[i] = padded_strides[i];
		else if (padded_shape[i] == 1) 
			new_strides[i] = 0;
		else
			throw std::runtime_error("broadcast_to: shape mismatch at dim " + std::to_string(i));
	}

	auto view = std::make_shared<TensorView<T>>(
			target_shape,
			src.shared_data(),
			new_strides,
			src.get_offset()
			);

	Tensor<T> result(view);
	return result;
}


}
