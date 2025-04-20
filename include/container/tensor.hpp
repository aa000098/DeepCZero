#pragma once

#include "container/tensorview.hpp"

#include <cstddef>
#include <vector>

namespace tensor {
	template<typename T>
	class Tensor {
	private:
		std::vector<size_t> shape;
		std::shared_ptr<std::vector<T>> data_ptr;
		std::vector<size_t> strides;

	public:
		Tensor(	const std::vector<size_t>& shape, 
				T init = T());

		Tensor(	const std::vector<size_t>& shape, 
				const std::vector<T>& init_data);

		Tensor() = default;

		TensorView<T> operator[](size_t idx);
		const tensor::TensorView<T> operator[](size_t idx) const;
		
		T& operator()(const std::vector<size_t>& indices) { return (*data_ptr)[flatten_index(indices)]; };
		const T& operator()(const std::vector<size_t>& indices) const { return (*data_ptr)[flatten_index(indices)]; };
		
		std::vector<float>& raw_data() {return *data_ptr;};
		const std::vector<size_t>& get_shape() const { return shape; };
		bool empty() { return (*data_ptr).empty(); };
		size_t size() const {return (*data_ptr).size(); };
		size_t ndim() const {return shape.size(); };

	private:
		void compute_strides();
		size_t flatten_index(const std::vector<size_t>& indices);
	};
}

#include "container/tensor.tpp"
