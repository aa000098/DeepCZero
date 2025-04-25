#pragma once

#include <memory>
#include <vector>
#include <stdexcept>

#include <iostream>

namespace tensor {
	template<typename T>
	class TensorView {
	private:
		std::vector<size_t> shape;
		std::shared_ptr<std::vector<T>> data_ptr;
		std::vector<size_t> strides;
		size_t offset;

	public:
		TensorView(	const std::vector<size_t>& shape, 
					std::shared_ptr<std::vector<T>> data_ptr, 
					const std::vector<size_t>& strides, 
					size_t offset = 0)
			: shape(shape), data_ptr(data_ptr), strides(strides), offset(offset) {};

		std::vector<T>& raw_data() { return (*data_ptr); };
		const std::vector<T>& raw_data() const { return (*data_ptr); };
		const std::vector<size_t>& get_shape() const {return shape; };
		const std::vector<size_t>& get_strides() const {return strides; };

		size_t ndim() const { return shape.size(); };
		size_t size() const;
		
		T& operator()(const std::vector<size_t>& indices);

		const T& operator()(const std::vector<size_t>& indices) const { return const_cast<TensorView*> (this)->operator()(indices); };

		TensorView<T> operator[](size_t index) const;
/*
		friend TensorView<T>& operator+=(TensorView<T>& lhs, const TensorView<T>& rhs);
		
		friend T operator*(TensorView<T>& a, size_t n);
		friend TensorView<T>& operator+(TensorView<T>& a, const TensorView<T>& b);
*/
		TensorView<T>& exp();
		TensorView<T>& pow(size_t mul);

//		friend std::ostream& operator<<(std::ostream& os, const TensorView<T>& view);
//
		void show();
	};
}

#include "container/tensor/tensorview.tpp"
