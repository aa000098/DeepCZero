#pragma once

#include "container/tensor/tensorbase.hpp"

#include <memory>
#include <vector>
#include <stdexcept>

namespace tensor {

	template<typename T>
	class TensorView : public TensorBase<T> {
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

		std::shared_ptr<std::vector<T>> shared_data() const override {return data_ptr; };
		std::vector<T>& raw_data() override { return (*data_ptr); };
		const std::vector<T>& raw_data() const override { return (*data_ptr); };
		std::vector<size_t> get_shape() const override {return shape; };
		std::vector<size_t> get_strides() const {return strides; };

		size_t ndim() const override { return shape.size(); };
		size_t size() const override ;
		size_t get_offset() const override { return offset; };
		bool empty() const { return (*data_ptr).empty(); };
		
		T& operator()(const std::vector<size_t>& indices) override;

		const T& operator()(const std::vector<size_t>& indices) const { 
			return const_cast<TensorView*> (this)->operator()(indices); };

		std::shared_ptr<TensorBase<T>> slice(size_t dim, size_t start, size_t end) const; 

		TensorView<T> operator[](size_t idx) const;

//		friend std::ostream& operator<<(std::ostream& os, const TensorView<T>& view);

		void show() const override;
	};
}

#include "container/tensor/tensorview.tpp"
