#include "container/tensorview.hpp"

#include <stdexcept>

namespace tensor {
	template<typename T>
	TensorView<T> Tensor<T>::operator[](size_t idx) {
		if (shape.size() < 2)
			throw std::runtime_error("Tensor must be at least 2D to support [] access returning view");

		std::vector<size_t> new_shape(shape.begin() + 1, shape.end());
		std::vector<size_t> new_strides(strides.begin() + 1, strides.end());
		size_t offset = idx * strides[0];

		return TensorView<T>(new_shape, new_strides, data_ptr, offset);
	}

	template<typename T>
	const TensorView<T> Tensor<T>::operator[](size_t idx) const {
		if (shape.size() < 2)
			throw std::runtime_error("Tensor must be at least 2D to support [] access returning view");

		std::vector<size_t> new_shape(shape.begin() + 1, shape.end());
		std::vector<size_t> new_strides(strides.begin() + 1, strides.end());
		size_t offset = idx * strides[0];

		return TensorView<T>(new_shape, new_strides, data_ptr, offset);
	}

	template<typename T>
	Tensor<T>::Tensor(	const std::vector<size_t>& shape, 
						T init) 
	: shape(shape) {
		size_t total_size = 1;
		for (auto dim : shape) total_size *= dim;
		data_ptr = std::make_shared<std::vector<T>>(total_size, init);
		compute_strides();
	}

	template<typename T>
	Tensor<T>::Tensor(	const std::vector<size_t>& shape, 
						const std::vector<T>& init_data) 
	: shape(shape) {
		data_ptr = std::make_shared<std::vector<T>>(init_data);
		compute_strides();
	}

	template<typename T>
	void Tensor<T>::compute_strides() {
		strides.resize(shape.size());
		size_t stride = 1;
		for (int i = shape.size() - 1; i >= 0; --i) {
			strides[i] = stride;
			stride *= shape[i];
		}
	}

	template<typename T>
	size_t Tensor<T>::flatten_index(const std::vector<size_t>& indices) {
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
}
