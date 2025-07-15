#pragma once

#include "container/tensor/tensorbase.hpp"
#include "container/tensor/tensor1D.hpp"
#include "container/tensor/tensorND.hpp"

#include <memory>

namespace tensor {
	template<typename T = float>
	class Tensor {
	private:
		std::shared_ptr<TensorBase<T>> impl;

	public:
		Tensor() : impl(std::make_shared<TensorND<T>>()) {};

		explicit Tensor(std::shared_ptr<TensorBase<T>> ptr) 
			: impl(std::move(ptr)) {};

		Tensor(size_t len, T init = T())
			: impl(std::make_shared<Tensor1D<T>>(len, init)) {};

		Tensor(	const std::vector<size_t>& shape, 
				T init = T{}) {
			if (shape.size() == 0)
				impl = std::make_shared<Tensor1D<T>>(std::vector<T>{init});
			else if(shape.size() == 1)
				impl = std::make_shared<Tensor1D<T>>(shape[0], init);
			else
				impl = std::make_shared<TensorND<T>>(shape, init);
		};
		
		Tensor(	const std::initializer_list<size_t>& shape, 
				T init = T{}) {
			*this = Tensor(std::vector<size_t>(shape), init);
		};

		Tensor(	const std::vector<size_t>& shape,
				const std::vector<T>& init) {
			if (shape.size() == 0)
				impl = std::make_shared<Tensor1D<T>>(std::vector<T>{init});
			else if (shape.size() == 1)
				impl = std::make_shared<Tensor1D<T>>(init);
			else
				impl = std::make_shared<TensorND<T>>(shape, init);
		};

// arithmetic operators
		Tensor<T>& operator+=(const Tensor<T>& other) {
			add_inplace(*this, other);
			return *this;
		}

		Tensor<T>& operator-=(const Tensor<T>& other) {
			sub_inplace(*this, other);
			return *this;
		}

		Tensor<T>& operator*=(const Tensor<T>& other) {
			mul_inplace(*this, other);
			return *this;
		}

		Tensor<T> operator+=(T scalar) {
			add_scalar_inplace(*this, scalar);
			return *this;
		}

		Tensor<T> operator-=(T scalar) {
			sub_scalar_inplace(*this, scalar);
			return *this;
		}

		Tensor<T>& operator*=(T scalar) {
			mul_scalar_inplace(*this, scalar);
			return *this;
		}

		Tensor<T>& operator/=(T scalar) {
			div_scalar_inplace(*this, scalar);
			return *this;
		}

// slicing operators
		Tensor<T> operator[](size_t idx) const {
			return Tensor<T>((*impl)[idx]); };
		
		T& operator()(const std::vector<size_t>& indices) {
			return (*impl)(indices); };

		const T& operator()(const std::vector<size_t>& indices) const {
			return (*impl)(indices); };
	
		Tensor<T> slice(size_t dim, size_t start, size_t end) {
			std::shared_ptr<TensorBase<T>> sliced_tensor = impl->slice(dim, start, end);
			return Tensor<T>(sliced_tensor);
		};

		Tensor<T> slice(const std::vector<std::pair<size_t, size_t>>& ranges) const {
			if (ranges.size() > this->ndim())
				throw std::runtime_error("Too many slicing dimensions");

			Tensor<T> result = *this;
			for (size_t dim = 0; dim < ranges.size(); ++dim) {
				auto [start, end] = ranges[dim];
				result = result.slice(dim, start, end);
			}
			return result;
		}

		Tensor<T> gather_rows(const std::vector<size_t>& indices) const { 
			return Tensor<T>(impl->gather_rows(indices)); };
		Tensor<T> gather_rows(const Tensor<size_t>& indices) const { 
			return this->gather_rows(indices.raw_data()); };

		Tensor<T>& operator=(const TensorView<T>& view) {
			this->impl = std::make_shared<TensorView<T>>(view);
			return *this;
		}

// common functions
		TensorView<T> view() const;
		std::vector<size_t> get_shape() const { 
			return impl->get_shape(); };
		std::vector<size_t> get_strides() const {
			return impl->get_strides(); };
		size_t get_offset() const {
			return impl->get_offset(); };
		size_t size() const {
			return impl->size(); };
		size_t ndim() const {
			return impl->get_shape().size(); };
		bool empty() const { 
			if (!impl) return true;
			else return impl->empty(); };
		std::vector<T> data() const;

		std::shared_ptr<std::vector<T>> shared_data() const {
			return impl->shared_data(); };
		std::vector<T>& raw_data() {
			return impl->raw_data(); };
		const std::vector<T>& raw_data() const {
			return impl->raw_data(); };

		std::vector<T> view_data() const;

// Tensor functions
		Tensor<T> clone() const {
			return Tensor<T>(this->get_shape(), this->raw_data()); };
		Tensor<T> reshape_like(const Tensor<T>& other) const;
		Tensor<T> reshape(const std::vector<size_t>& new_shape) const;
		Tensor<T> transpose(const std::vector<size_t>& axes={}) const;
		Tensor<T> sum(const std::vector<int>& axis = {},
						 bool keepdims = false) const;
		Tensor<T> max(const std::vector<int>& axes,
						bool keepdims) const;
		Tensor<size_t> argmax(int axis, 
								bool keepdims = false) const;
		Tensor<uint8_t> equal(const Tensor<T>& other) const; 
		float mean() const;
	
		Tensor<T> pad(const std::vector<std::pair<size_t, size_t>>& padding, T pad_value) const;

		Tensor<T> contiguous() const;

// gemm functions
		Tensor<T>& dot(const Tensor<T>& other) {
			return dot(*this, other);
		}

// io functions
		void to_csv(const std::string& filename, 
					bool index = false, 
					bool header = false, 
					char delimiter = ',');

		static Tensor<T> from_csv(	const std::string& filename, 
							bool index = false,
							bool header = false,
							char delimiter = ',');

		void show() const {
			if (!impl) std::cout << "[  ]" << std::endl;
			else impl->show(); };
	};
}

#include "container/tensor/tensor.tpp"
