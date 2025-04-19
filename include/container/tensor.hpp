#pragma once

#include <cstddef>
#include <vector>

namespace tensor {
	template<typename T>
	class Tensor {
	private:
		std::vector<size_t> shape;
		std::vector<T> data;
		std::vector<size_t> strides;

	public:
		Tensor(const std::vector<size_t>& shape, T init = T());
		Tensor(const std::vector<size_t>& shape, const std::vector<T>& init_data);

		Tensor() = default;

		T& operator[](size_t idx) {return data[idx]; };
		const T& operator[](size_t idx) const {return data[idx]; };
		T& operator()(const std::vector<size_t>& indices) { return data[flatten_index(indices)]; };

		const T& operator()(const std::vector<size_t>& indices) const { return data[flatten_index(indices)]; };
		const std::vector<float>& raw_data() const {return data;};
		const std::vector<size_t>& get_shape() const { return shape; };
		bool empty() { return data.empty(); };
		size_t size() const {return data.size(); };
		size_t ndim() const {return shape.size(); };

	private:
		void compute_strides();
		size_t flatten_index(const std::vector<size_t>& indices);
	};
}

#include "container/tensor.tpp"
