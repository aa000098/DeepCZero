#pragma once

#include "container/tensorview.hpp"

#include <cstddef>
#include <vector>

namespace tensor {
	template<typename T>
	class TensorND : public TensorBase {
	private:
		std::shared_ptr<std::vector<T>> data_ptr;
		std::vector<size_t> shape;
		std::vector<size_t> strides;
		size_ offset = 0;

	public:
		TensorND(	const std::vector<size_t>& shape, 
					T init = T());

		TensorND(	const std::vector<size_t>& shape, 
					const std::vector<T>& init_data);

		TensorND() = default;

		TensorView<T> operator[](size_t idx);
		const tensor::TensorView<T> operator[](size_t idx) const;
		
		T& operator()(	
				const std::vector<size_t>& indices) {
			return (*data_ptr)[flatten_index(indices)]; };
		const T& operator()(
				const std::vector<size_t>& indices) const { 
			return (*data_ptr)[flatten_index(indices)]; };
		
		std::vector<float>& raw_data() {
			return *data_ptr;};
		const std::vector<float>& raw_data() const {
			return *data_ptr;};

		const std::vector<size_t>& get_shape() const { 
			return shape; };

		size_t size() const override {
			return (*data_ptr).size(); };
		size_t ndim() const override {
			return shape.size(); };
		bool empty() const { 
			return (*data_ptr).empty(); };

	private:
		void compute_strides();
		size_t flatten_index(const std::vector<size_t>& indices);
	};
}

#include "container/tensorND.tpp"
