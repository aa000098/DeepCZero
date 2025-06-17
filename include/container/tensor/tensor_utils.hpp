#pragma once

#include <numeric>
#include <set>

namespace tensor {

inline size_t product(const std::vector<size_t>& shape) {
    return std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>());
}

inline std::vector<size_t> unflatten_index(size_t flat_index, const std::vector<size_t>& shape) {
    std::vector<size_t> indices(shape.size());
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        indices[i] = flat_index % shape[i];
        flat_index /= shape[i];
    }
    return indices;
}

inline size_t flatten_index(const std::vector<size_t>& indices, const std::vector<size_t>& strides) {
	size_t flat = 0;
	for (size_t i = 0; i < indices.size(); ++i) {
		flat += indices[i] * strides[i];
	}
	return flat;
}

inline std::set<size_t> normalize_axes(const std::vector<int>& axes, size_t ndim) {
    std::set<size_t> result;
    for (int a : axes) {
        if (a < 0) result.insert(static_cast<size_t>(ndim + a));
        else result.insert(static_cast<size_t>(a));
    }
    return result;
}

inline std::vector<size_t> compute_reduced_shape(
    const std::vector<size_t>& shape, 
    const std::set<size_t>& reduce_axes, 
    bool keepdims) 
{
    std::vector<size_t> result;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (reduce_axes.count(i)) {
            if (keepdims) result.push_back(1);
        } else {
            result.push_back(shape[i]);
        }
    }
    return result;
}


inline std::vector<size_t> compute_contiguous_strides(const std::vector<size_t>& shape) {
    std::vector<size_t> strides(shape.size());
    if (!shape.empty()) {
        strides.back() = 1;
        for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }
    return strides;
}

}
