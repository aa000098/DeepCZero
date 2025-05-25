#pragma once
#include "container/tensor/tensor_all.hpp"

#include <random>

using namespace tensor;

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

template<typename T>
inline std::tuple<Tensor<T>, Tensor<T>, std::vector<size_t>> broadcast_binary_operands(
    const Tensor<T>& a, const Tensor<T>& b) {

	std::vector<size_t> shape_a = a.get_shape();
	std::vector<size_t> shape_b = b.get_shape();
	size_t ndim = std::max(shape_a.size(), shape_b.size());

	while (shape_a.size() < ndim) shape_a.insert(shape_a.begin(), 1);
	while (shape_b.size() < ndim) shape_b.insert(shape_b.begin(), 1);

	std::vector<size_t> broadcast_shape(ndim);
	for (size_t i = 0; i < ndim; ++i) {
		if (shape_a[i] == shape_b[i])
			broadcast_shape[i] = shape_a[i];
		else if (shape_a[i] == 1)
			broadcast_shape[i] = shape_b[i];
		else if (shape_b[i] == 1)
			broadcast_shape[i] = shape_a[i];
		else
			throw std::runtime_error("broadcast_binary_operands: shape mismatch");
	}

	Tensor<T> a_bc = broadcast_to(a, broadcast_shape);
	Tensor<T> b_bc = broadcast_to(b, broadcast_shape);

	return {a_bc, b_bc, broadcast_shape};
}


inline Tensor<> rand_tensor(size_t rows, size_t cols, size_t seed) {
	std::mt19937 gen(seed);

	std::uniform_real_distribution<> dist(-1, 1);

	std::vector<float> data;
	data.reserve(rows * cols);
	for (size_t i = 0; i < rows * cols; i++)
		data.push_back(dist(gen));

	return Tensor<>({rows, cols}, data);
}

inline Tensor<> rand_tensor(size_t rows, size_t cols) {
	std::random_device rd;
	return rand_tensor(rows, cols, rd());
}
