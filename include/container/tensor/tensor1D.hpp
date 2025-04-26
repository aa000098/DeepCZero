#pragma once

#include "container/tensor/tensorbase.hpp"

#include <iostream>

namespace tensor {
	template<typename T>
	class Tensor1D : public TensorBase<T> {
	private:
		std::vector<T> data;
	public:
		Tensor1D(const std::vector<T>& vec) : data(vec) {};
		Tensor1D(size_t len, T init = T()) : data(len, init) {};

		T& operator()(const std::vector<size_t>& indices) override {
			return data[indices[0]];
		};
		std::vector<size_t> get_shape() const { return {data.size()}; };
		size_t size() const override { return data.size(); };
		size_t ndim() const override { return 1; };
		bool empty() const override { 
			return data.empty(); };

		void show() const override {
			std::cout << "[ ";
			for (size_t i = 0; i < data.size(); i++) {
				std::cout << data[i];
				if (i != data.size() - 1) 
					std::cout << ", ";
			}
			std::cout << " ]" << std::endl;
		};

	};
}
