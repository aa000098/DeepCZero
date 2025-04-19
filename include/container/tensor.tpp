#include "container/tensor.hpp"

#include <stdexcept>

template<typename T>
tensor::Tensor<T>::Tensor(const std::vector<size_t>& shape, T init) : shape(shape) {
	size_t total_size = 1;
	for (auto dim : shape) total_size *= dim;
	data = std::vector<T>(total_size, init);
	compute_strides();
}

template<typename T>
tensor::Tensor<T>::Tensor(const std::vector<size_t>& shape, const std::vector<T>& init_data) : shape(shape), data(init_data) {
	compute_strides();
}

template<typename T>
void tensor::Tensor<T>::compute_strides() {
	strides.resize(shape.size());
	size_t stride = 1;
	for (int i = shape.size() - 1; i >= 0; --i) {
		strides[i] = stride;
		stride *= shape[i];
	}
}

template<typename T>
size_t tensor::Tensor<T>::flatten_index(const std::vector<size_t>& indices) {
	if (indices.size() != shape.size())
		throw std::invalid_argument("Dimension mismatch in %s", __func__);
	size_t idx = 0;
	for (size_t i = 0; i < indices.size(); ++i) {
		if (indices[i] >= shape[i])
			throw std::out_of_range("Index out of bounds");
		idx += indices[i] * strides[i];
	}
	return idx;
}

