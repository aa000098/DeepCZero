#pragma once

#include "container/tensor/tensorbase.hpp"

#include <iostream>

namespace tensor {
	template<typename T>
	class Tensor1D : public TensorBase<T> {
	private:
		std::shared_ptr<std::vector<T>> data_ptr;
	public:
		Tensor1D(const std::vector<T>& vec) : data_ptr(std::make_shared<std::vector<T>>(vec)) {};
		Tensor1D(size_t len, T init = T()) : data_ptr(std::make_shared<std::vector<T>>(len, init)) {};
		Tensor1D() : data_ptr(std::make_shared<std::vector<T>>()) {};

		T& operator()(const std::vector<size_t>& indices) override {
			return (*data_ptr)[indices[0]];
		};

		std::shared_ptr<TensorBase<T>> slice(size_t dim, size_t start, size_t end) const {
			if (dim != 0)
				throw std::invalid_argument("Tensor1D only supports slicing along dimension 0.");
			if (start >= end || end > data_ptr->size())
				throw std::out_of_range("Invalid slice range for Tensor1D");

			std::vector<T> sliced(data_ptr->begin() + start, data_ptr->begin() + end);
			return std::make_shared<Tensor1D<T>>(sliced);
		}; 

		std::vector<size_t> get_shape() const override { return {data_ptr->size()}; };
		std::vector<size_t> get_strides() const override {
			return {1}; };
		size_t size() const override { return data_ptr->size(); };
		size_t ndim() const override { return 1; };
		size_t get_offset() const override { return 0; };
		bool empty() const override { 
			return data_ptr->empty(); };

		std::vector<T>& raw_data() override {
			return *data_ptr;};
		const std::vector<T>& raw_data() const override {
			return *data_ptr;};
		std::shared_ptr<std::vector<T>> shared_data() const override {
			return data_ptr;
		};
	
		void show() const override {
			std::cout << "[ ";
			for (size_t i = 0; i < data_ptr->size(); i++) {
				std::cout << (*data_ptr)[i];
				if (i != data_ptr->size() - 1) 
					std::cout << ", ";
			}
			std::cout << " ]" << std::endl;
		};

	};
}
