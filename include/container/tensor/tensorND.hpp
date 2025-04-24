#pragma once

#include "container/tensor/tensorbase.hpp"
#include "container/tensor/tensorview.hpp"

#include <cstddef>
#include <vector>

namespace tensor {
	template<typename T>
	class TensorND : public TensorBase<T> {
	private:
		std::vector<size_t> shape;
		std::vector<T> data;
		std::vector<size_t> strides;
		size_t offset = 0;

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
			return data[flatten_index(indices)]; };
		const T& operator()(
				const std::vector<size_t>& indices) const { 
			return data[flatten_index(indices)]; };
		
		std::vector<float>& raw_data() {
			return data;};
		const std::vector<float>& raw_data() const {
			return data;};


		
		std::vector<size_t> get_shape() const { 
			return shape; };
		std::vector<size_t> get_strides() const {
			return strides; };
		size_t size() const override {
			return data.size(); };
		size_t ndim() const override {
			return shape.size(); };
		bool empty() const override { 
			return data.empty(); };

		void show();

	private:
		void compute_strides();
		size_t flatten_index(const std::vector<size_t>& indices);
	};
}

#include "container/tensor/tensorND.tpp"
