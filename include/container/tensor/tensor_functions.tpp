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


template<typename T>
Tensor<T> sum_to(const Tensor<T>& src, const std::vector<size_t>& target_shape) {

	const auto& src_shape = src.get_shape();

	size_t ndim_src = src_shape.size();
	size_t ndim_target = target_shape.size();

	if (ndim_src < ndim_target)
		std::runtime_error("sum_to: target shape has more dimensions than source.");

	std::vector<int> reduce_axes;
	size_t leading = ndim_src - ndim_target;

	for (size_t i = 0; i < leading; i++)
		reduce_axes.push_back(static_cast<int>(i));

	for (size_t i = 0; i < ndim_target; i++) {
		size_t src_dim = src_shape[leading + i];
		size_t tgt_dim = target_shape[i];
		if (tgt_dim == 1 && src_dim != 1) 
			reduce_axes.push_back(static_cast<int>(leading + i));
		else if (tgt_dim != src_dim && tgt_dim != 1)
			throw std::runtime_error("sum_to: shapes are not broadcast-compatible.");
	}
	
	if (reduce_axes.empty()) 
		return src;

	Tensor<T> summed = src.sum(reduce_axes, true);

	if (summed.get_shape() != target_shape)
		summed = summed.reshape(target_shape);

	return summed;

}

template <typename T>
void add_at(Tensor<T>& gx, 
			const std::vector<size_t>& slices, 
			const Tensor<T>& gy) {

	TensorView<T> view = gx.view();
	for (size_t idx : slices)
		view = view[idx];

	auto& gx_data = view.raw_data();
	const auto& gy_data = gy.raw_data();
	size_t offset = view.get_offset();

	for (size_t i = 0; i < gy.size(); ++i)
		gx_data[offset + i] += gy_data[i]; 

}


}
