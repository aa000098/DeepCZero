#pragma once

#include "container/tensor/tensorbase.hpp"
#include "container/tensor/tensorview.hpp"


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

		std::shared_ptr<TensorBase<T>> operator[](size_t index) override {
			if (index >= this->size())
				throw std::out_of_range("Tensor1D::operator[] index out of range");
			std::vector<T> v = { (*data_ptr)[index] };
			return std::make_shared<Tensor1D<T>>(v);
		}

		// TensorBase override
		T& operator()(const std::vector<size_t>& indices) override {
			return (*data_ptr)[indices[0]];
		};

		std::shared_ptr<TensorBase<T>> slice(size_t dim, size_t start, size_t end) const override;

		std::shared_ptr<TensorBase<T>> gather_rows(const std::vector<size_t>& indices) const;

		std::vector<size_t> get_shape() const override { return {data_ptr->size()}; };
		std::vector<size_t> get_strides() const override {
			return {1}; };
		size_t size() const override { return data_ptr->size(); };
		size_t ndim() const override { return 1; };
		size_t get_offset() const override { return 0; };
		bool empty() const override { 
			return data_ptr->empty(); };

		std::vector<T>& raw_data() override {
			return *data_ptr; };
		const std::vector<T>& raw_data() const override {
			return *data_ptr; };
		std::shared_ptr<std::vector<T>> shared_data() const override {
			return data_ptr; };
	
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

#include "container/tensor/tensor1D.tpp"
