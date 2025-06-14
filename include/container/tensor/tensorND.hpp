#pragma once

#include "container/tensor/tensorbase.hpp"
#include "container/tensor/tensorview.hpp"
#include "container/tensor/tensor_utils.hpp"

#include <cstddef>
#include <vector>
#include <memory>



namespace tensor {
	

	template<typename T>
	class TensorND : public TensorBase<T> {
	private:
		std::vector<size_t> shape;
		std::shared_ptr<std::vector<T>> data_ptr;
		std::vector<size_t> strides;
		size_t offset = 0;

	public:
		TensorND(	const std::vector<size_t>& shape, 
					T init = T{});

		TensorND(	const std::vector<size_t>& shape, 
					const std::vector<T>& init_data);

		TensorND() : data_ptr(std::make_shared<std::vector<T>>()) {};

		TensorView<T> operator[](size_t idx);
		const TensorView<T> operator[](size_t idx) const;
		
		T& operator()(	
				const std::vector<size_t>& indices) {
			return (*data_ptr)[flatten_index(indices, strides)]; };
		const T& operator()(
				const std::vector<size_t>& indices) const { 
			return (*data_ptr)[flatten_index(indices, strides)]; };
		std::shared_ptr<TensorBase<T>> slice(size_t dim, size_t start, size_t end) const override ;

		std::shared_ptr<TensorBase<T>> gather_rows(const std::vector<size_t>& indices) const override;
		
		TensorView<T> view(size_t index) const;
		
		std::vector<size_t> get_strides() const override {
			return strides; };

		std::shared_ptr<std::vector<T>> shared_data() const {
			return data_ptr; };

		// TensorBase override
		std::vector<size_t> get_shape() const override { 
			return shape; };
		size_t size() const override {
			return (*data_ptr).size(); };
		size_t ndim() const override {
			return shape.size(); };
		size_t get_offset() const override {
			return offset; };
		bool empty() const override { 
			return (*data_ptr).empty(); };
		std::vector<T>& raw_data() override {
			return *data_ptr;};
		const std::vector<T>& raw_data() const override {
			return *data_ptr;};
	
		void show() const override;

	private:
		void compute_strides();
//		size_t flatten_index(const std::vector<size_t>& indices);
	};
}

#include "container/tensor/tensorND.tpp"
