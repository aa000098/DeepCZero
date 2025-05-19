#pragma once
#include "container/tensor/tensor_all.hpp"

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

// 범용 flatten_index (전역 함수, tensor/utils.hpp)
inline size_t flatten_index(const std::vector<size_t>& indices, const std::vector<size_t>& strides) {
	size_t flat = 0;
	for (size_t i = 0; i < indices.size(); ++i) {
		flat += indices[i] * strides[i];
	}
	return flat;
}


