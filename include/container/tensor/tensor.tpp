#pragma once

#include "container/tensor/tensor.hpp"
#include "container/tensor/tensor_utils.hpp"

#include <set>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <filesystem>
#include <functional>

namespace tensor {

template<typename T>
TensorView<T> Tensor<T>::view() const {
	auto* ptr = dynamic_cast<TensorND<T>*>(impl.get());
	if (!ptr)
		throw std::runtime_error("Only TensorND supports view()");

	return TensorView<T>(
			ptr->get_shape,
			ptr->shared_data(),
			ptr->get_strides(),
			ptr->get_offset()
			);
}

template<typename T>
Tensor<T> Tensor<T>::reshape_like(const Tensor<T>& other) const {
	const auto& target_shape = other.get_shape();
	size_t target_size = other.size();

	//this->show();
	//other.show();
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
Tensor<T> Tensor<T>::max(const std::vector<int>& axes,
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

template <typename T>
Tensor<size_t> argmax_along_axis(const Tensor<T>& x, int axis, bool keepdims) {
    const auto& src_shape = x.get_shape();
	const auto& src_stride = x.get_strides();
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

	Tensor<size_t> result(result_shape, 0);
	auto& result_data = result.raw_data();
	std::vector<bool> is_initialized(result_data.size(), false);

	// 2. 순회하며 argmax 계산
	size_t total = x.size();
	for (size_t flat_idx = 0; flat_idx < total; flat_idx++) {
		std::vector<size_t> idx = unflatten_index(flat_idx, src_shape);

		// argmax result 인덱스 
		std::vector<size_t> dst_idx;
		for (size_t i = 0; i < ndim; i++) {
			if (i == static_cast<size_t>(axis)) {
				if (keepdims) dst_idx.push_back(0);
				continue;
			}
			dst_idx.push_back(idx[i]);
		}

		size_t dst_flat = flatten_index(dst_idx, result_strides);

		// 타겟 axis에서의 flat_idx의 unflatten index  
		size_t candidate_idx = idx[axis];

		// 첫 초기화
		if (!is_initialized[dst_flat]) {
			result_data[dst_flat] = candidate_idx;
			is_initialized[dst_flat] = true;
		} else { 
		// 임시 최대값 result_data[dst_flat]과 현재값 src_data[flat_idx] 비교 
			std::vector<size_t> old_idx = replace_axis(idx, axis, result_data[dst_flat]);		
			if (src_data[flat_idx] > src_data[flatten_index(old_idx, src_stride)])
				result_data[dst_flat] = candidate_idx;
		}

	}

	return result;
}

template <typename T>
Tensor<size_t> Tensor<T>::argmax(int axis, 
								bool keepdims) const {
	return argmax_along_axis(*this, axis, keepdims);
}


template <typename T>
Tensor<uint8_t> Tensor<T>::equal(const Tensor<T>& other) const {
	const auto& a = this->raw_data();
	const auto& b = other.raw_data();

	if (this->get_shape() != other.get_shape())
		throw std::invalid_argument("equal: shape mismatch");

	std::vector<uint8_t> result_data(a.size());
	for (size_t i = 0; i < a.size(); i++)
		result_data[i] = (a[i] == b[i]);

	return Tensor<uint8_t>(this->get_shape(), result_data);
}


template <typename T>
float Tensor<T>::mean() const {
	const auto& data = this->raw_data();

	if (data.empty())
		throw std::runtime_error("mean(): tensor is empty");

	float sum = this->sum()({0});

	return sum / static_cast<float>(data.size());
}

template <typename T>
std::vector<T> Tensor<T>::data() const {
	if (auto view = dynamic_cast<const TensorView<T>*>(impl.get()))
		return view_data();
	else
		return raw_data();
}

template <typename T>
std::vector<T> Tensor<T>::view_data() const {
	std::vector<T> result(size());

	auto shape = impl->get_shape();
	auto strides = impl->get_strides();
	auto offset = impl->get_offset();
	const auto& data = impl->raw_data();

	size_t ndim = shape.size();

	// index 상태 저장용
	std::vector<size_t> indices(ndim, 0);

	for (size_t i = 0; i < size(); ++i) {
		// linear offset 계산
		size_t linear_index = offset;
		for (size_t d = 0; d < ndim; ++d)
			linear_index += indices[d] * strides[d];

		result[i] = data[linear_index];

		// 다음 index 계산 (다중 루프 대신 1D 인덱스 -> N차원 인덱스로 변환)
		for (ssize_t d = ndim - 1; d >= 0; --d) {
			if (++indices[d] < shape[d]) break;
			indices[d] = 0;  // overflow → 다음 차원 증가
		}
	}

	return result;
}

template <typename T>
Tensor<T> Tensor<T>::pad(const std::vector<std::pair<size_t, size_t>>& padding, T pad_value) const {
	auto old_shape = this->get_shape();
	size_t ndim = old_shape.size();

	std::vector<size_t> new_shape(ndim);
	for (size_t i = 0; i < ndim; i++)
		new_shape[i] = old_shape[i] + padding[i].first + padding[i].second;

	Tensor<T> padded_tensor(new_shape, pad_value);

	std::vector<size_t> old_idx(ndim, 0);
	std::vector<size_t> new_idx(ndim, 0);

	std::function<void(size_t)> copy_data = [&](size_t dim) {
		if (dim == ndim) {
			padded_tensor(new_idx) = (*this)(old_idx);
			return;
		}
		for (size_t i = 0; i < old_shape[dim]; i++) {
			old_idx[dim] = i;
			new_idx[dim] = i + padding[dim].first;
			copy_data(dim + 1);
		}
	};

	copy_data(0);

	return padded_tensor;
}

template<typename T>
Tensor<T> Tensor<T>::contiguous() const {
    Tensor<T> result(get_shape());
    std::vector<T>& result_data = result.raw_data();
    std::vector<size_t> shape = get_shape();
    std::vector<size_t> indices(shape.size(), 0);

    size_t total = impl->size();
    for (size_t i = 0; i < total; ++i) {
        result_data[i] = (*this)(indices);
        for (size_t j = shape.size() - 1; j < shape.size(); --j) {
            if (++indices[j] < shape[j]) break;
            indices[j] = 0;
            if (j == 0) break;
        }
    }

    return result;
}

template<typename T>
Tensor<T> Tensor<T>::ravel() const {
	if (this->ndim() <= 1) return *this;

	return this->reshape({this->size()});
}

template<typename T>
Tensor<T> Tensor<T>::flatten() const {
	return Tensor<T>({this->size()}, this->raw_data());
}

template<typename T>
void Tensor<T>::to_csv(const std::string& filename, bool index, bool header, char delimiter) {
	// 파일 기본 저장 경로 설정
	const char* home = std::getenv("HOME");
	if (home == nullptr) 
		throw std::runtime_error("HOME environment variable not set.");

	std::string dir_path = std::string(home) + "/.deepczero/datasets/";
	std::filesystem::create_directories(dir_path);

	std::string full_path = dir_path + "/" + filename;
    std::ofstream out(full_path);
    if (!out.is_open())
		throw std::runtime_error("Failed to open file: " + full_path);

	const auto& shape = this->get_shape();
	size_t ndim = shape.size();

	if (ndim >= 3)
		throw std::runtime_error("to_csv only supports tensors with ndim <= 2");

    // 헤더 작성
	if (header) {
		if (index) {
        	out << "row" << delimiter;
		}
		if (ndim == 1)
    		out << "col\n";
		else if (ndim == 2) {
			for (size_t j = 0; j < shape[1]; j++) {
				out << "col" << j;
				if (j < shape[1] - 1)
					out << delimiter;
			}
			out << "\n";
		}
	}

    // 데이터 작성
	if (ndim == 1) {
		for (size_t i = 0; i < shape[0]; i++) {
			if (index) out << i << delimiter;
			out << std::setprecision(7) << this->raw_data()[i] << "\n";
		}
	} else if (ndim == 2) {
		size_t rows = shape[0];
		size_t cols = shape[1];
		for (size_t i = 0; i < rows; i++) {
			if (index) out << i << delimiter;
			for (size_t j = 0; j < cols; j++) {
				out << std::setprecision(7) << this->raw_data()[i * cols + j];
				if (j < cols - 1)
					out << delimiter;
			}
			out << "\n";
		}
	}

    out.close();
}

template<typename T>
Tensor<T> Tensor<T>::from_csv(const std::string& filename, bool index, bool header, char delimiter) {
	const char* home = std::getenv("HOME");
	if (home == nullptr) 
		throw std::runtime_error("HOME environment variable not set.");

	std::string dir_path = std::string(home) + "/.deepczero/datasets/";
	std::filesystem::create_directories(dir_path);

	std::string full_path = dir_path + "/" + filename;
    std::ifstream in(full_path);

	if (!in.is_open())
        throw std::runtime_error("Failed to open file: " + full_path);

	std::string line;
	std::vector<T> values;
	size_t num_cols = 0;
	size_t num_rows = 0;

	while (std::getline(in, line)) {
		if (header) {
			header = false;
			continue;
		}

		std::stringstream ss(line);
		std::string cell;
		size_t col_count = 0;

		if (index) 
			std::getline(ss, cell, delimiter);
		
		while (std::getline(ss, cell, delimiter)) {
			std::stringstream val_ss(cell);
			T val;
			val_ss >> val;
			values.push_back(val);
			col_count++;
		}

		if (col_count == 0) continue; 

		if (num_cols == 0) num_cols = col_count;
		else if (col_count != num_cols)
			throw std::runtime_error("Inconsistent column size in CSV file");

		num_rows++;
	}
	std::vector<size_t> shape = (num_cols == 1) ? std::vector<size_t>{num_rows} : std::vector<size_t> {num_rows, num_cols};

	return Tensor<T>(shape, values);
}

}



