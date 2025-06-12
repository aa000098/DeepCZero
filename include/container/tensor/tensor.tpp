#pragma once

#include "container/tensor/tensor.hpp"
#include "container/tensor/tensor_utils.hpp"

#include <set>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <filesystem>

namespace tensor {

template<typename T>
TensorView<T> Tensor<T>::view() const {
	auto* ptr = dynamic_cast<TensorND<T>*>(impl.get());
	if (!ptr)
		throw std::runtime_error("Only TensorND supports view()");

	return TensorView<T>(
			ptr->get_shape(),
			ptr->shared_data(),
			ptr->get_strides(),
			ptr->get_offset()
			);
}

template<typename T>
Tensor<T> Tensor<T>::reshape_like(const Tensor<T>& other) const {
	const auto& target_shape = other.get_shape();
	size_t target_size = other.size();

	this->show();
	other.show();
	if (target_size != this->size()) {
		throw std::runtime_error("reshape_like failed: element count mismatch");
	}

	const auto& source_shape = this->get_shape();
	

	return Tensor<T>(target_shape, this->raw_data()[0]);
}


template<typename T>
Tensor<T> Tensor<T>::reshape(const std::vector<size_t>& new_shape) const {
	size_t old_size = impl->size();
	size_t new_size = 1;
	for (size_t d : new_shape) new_size *= d;

	if (old_size != new_size) 
		throw std::runtime_error("reshape error: size mismatch");

	auto data_ptr = impl->shared_data();
	size_t offset = impl->get_offset();

	auto new_strides = compute_contiguous_strides(new_shape);

	auto view_impl = std::make_shared<TensorView<T>>(new_shape, data_ptr, new_strides, offset);

	Tensor<T> result;
	result.impl = view_impl;
	
	return result;
}

template<typename T>
Tensor<T> Tensor<T>::transpose(const std::vector<size_t>& axes) const {
    std::vector<size_t> old_shape = impl->get_shape();
    std::vector<size_t> old_strides = impl->get_strides();

    std::vector<size_t> actual_axes = axes;
    if (actual_axes.empty()) {
		size_t ndim = old_shape.size();
		for (size_t i = 0; i < ndim; i++)
			actual_axes.push_back(ndim-1-i);
    }

    if (actual_axes.size() != old_shape.size())
        throw std::runtime_error("transpose: axes size mismatch");

    std::vector<size_t> new_shape(actual_axes.size());
    std::vector<size_t> new_strides(actual_axes.size());

    for (size_t i = 0; i < actual_axes.size(); ++i) {
        new_shape[i] = old_shape[actual_axes[i]];
        new_strides[i] = old_strides[actual_axes[i]];
    }

    auto view_impl = std::make_shared<TensorView<T>>(
        new_shape,
        impl->shared_data(),
        new_strides,
        impl->get_offset()
    );

    Tensor<T> result;
    result.impl = view_impl;
    return result;
}

template<typename T>
Tensor<T> Tensor<T>::sum(const std::vector<int>& axis, bool keepdims) const {

    const auto& src_shape = impl->get_shape();
    const auto& src_strides = impl->get_strides();
    const auto& src_data = impl->raw_data();

    std::set<size_t> reduce_axis = normalize_axes(axis, src_shape.size());
	
	if (reduce_axis.empty()) {
    	for (size_t i = 0; i < src_shape.size(); ++i)
        	reduce_axis.insert(i);
	}

    // 1. 결과 shape 계산
    std::vector<size_t> result_shape = compute_reduced_shape(src_shape, reduce_axis, keepdims);;

    Tensor<T> result(result_shape, T{});
    auto& result_data = result.raw_data();
    auto result_strides = compute_contiguous_strides(result_shape);

    // 2. 모든 인덱스를 순회하여 값을 더함
    size_t total = src_data.size();

    for (size_t flat_idx = 0; flat_idx < total; ++flat_idx) {
		// 다차원 인덱스 계산
        std::vector<size_t> idx = unflatten_index(flat_idx, src_shape);;

        // 결과 텐서 인덱스 계산 (축 제외)
        std::vector<size_t> dst_idx;
        for (size_t i = 0; i < idx.size(); ++i) {
            if (!reduce_axis.count(i)) {
                dst_idx.push_back(idx[i]);
            } else if (keepdims) {
                dst_idx.push_back(0);
            }
        }

        // flatten index 계산
        size_t dst_flat = flatten_index(dst_idx, result_strides);
        result_data[dst_flat] += src_data[flat_idx];
    }

    return result;

}

template<typename T>
Tensor<T> max_along_axis(const Tensor<T>& x, int axis, bool keepdims) {
    const auto& src_shape = x.get_shape();
    const auto& src_data = x.raw_data();
    size_t ndim = src_shape.size();

    // 음수 axis 처리
    if (axis < 0) axis += static_cast<int>(ndim);

    if (axis < 0 || axis >= static_cast<int>(ndim))
        throw std::invalid_argument("max_along_axis: invalid axis");

    // 1. reduced shape 계산
    std::set<size_t> reduce_axes = { static_cast<size_t>(axis) };
    std::vector<size_t> result_shape = compute_reduced_shape(src_shape, reduce_axes, keepdims);
    std::vector<size_t> result_strides = compute_contiguous_strides(result_shape);

	Tensor<T> result(result_shape, std::numeric_limits<T>::lowest());
	auto& result_data = result.raw_data();

	// 2. 순회하며 max 계산
	size_t total = x.size();
	for (size_t flat_idx = 0; flat_idx < total; ++flat_idx) {
		std::vector<size_t> idx = unflatten_index(flat_idx, src_shape);

		std::vector<size_t> dst_idx;
		for (size_t i = 0; i < ndim; ++i) {
			if (reduce_axes.count(i)) {
				if (keepdims) 
					dst_idx.push_back(0);
			} else
				dst_idx.push_back(idx[i]);
		}

        size_t dst_flat = flatten_index(dst_idx, result_strides);
        result_data[dst_flat] = std::max(result_data[dst_flat], src_data[flat_idx]);
    }

    return result;
}



template <typename T>
Tensor<T> Tensor<T>::max(const std::vector<int> axes,
						bool keepdims) const {
	Tensor<T> result = *this;
	std::vector<int> sorted_axes;

	if (axes.empty()) {
		// 모든 축을 대상으로 수행
		for (int i = 0; i < static_cast<int>(this->ndim()); ++i)
			sorted_axes.push_back(i);
	} else 
		sorted_axes = axes;

	std::sort(sorted_axes.begin(), 
				sorted_axes.end(),
				std::greater<int>());

	for (int axis : sorted_axes)
		result = max_along_axis(result, axis, keepdims);

	if (!keepdims && sorted_axes.size() == this->ndim())
		result = result.reshape({});

	return result;

}

template<typename T>
void Tensor<T>::to_csv(const std::string& filename, bool index, bool header, char delimiter) {
	const char* home = std::getenv("HOME");
	if (home == nullptr) 
		throw std::runtime_error("HOME environment variable not set.");

	std::string dir_path = std::string(home) + "/.deepczero/datasets/";
	std::filesystem::create_directories(dir_path);

	std::string full_path = dir_path + "/" + filename;
    std::ofstream out(full_path);
    if (!out.is_open())
        throw std::runtime_error("Failed to open file: " + full_path);

    // 헤더 작성
	if (header) {
    	for (size_t i = 0; i < this->size(); ++i) 
        	out << "dim" << i << delimiter;
    	out << "value\n";
	}

    // 데이터 작성
    for (size_t flat_idx = 0; flat_idx < this->size(); ++flat_idx) {
		if (index) {
        	std::vector<size_t> indices = unflatten_index(flat_idx, this->get_shape());
        	for (size_t i = 0; i < indices.size(); ++i)
            	out << indices[i] << delimiter;
		}
        out << std::setprecision(7) << this->raw_data()[flat_idx] << "\n";
    }

    out.close();
}

}



