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
		Tensor() = default;

		Tensor(std::shared_ptr<TensorBase<T>> ptr) 
			: impl(std::move(ptr)) {};

		Tensor(size_t len, T init = T())
			: impl(std::make_shared<Tensor1D<T>>(len, init)) {};

		Tensor(	const std::vector<size_t>& shape, 
				T init = T{}) {
			if (shape.size() == 1)
				impl = std::make_shared<Tensor1D<T>>(init);
			else
				impl = std::make_shared<TensorND<T>>(shape, T{});
		};

		Tensor(	const std::vector<size_t>& shape,
				const std::vector<T>& init) {
			if (shape.size() == 1)
				impl = std::make_shared<Tensor1D<T>>(init);
			else
				impl = std::make_shared<TensorND<T>>(shape, init);
		};

		TensorView<T> operator[](size_t idx) {
			auto* ptr = dynamic_cast<TensorND<T>*>(impl.get());
			if(!ptr) throw std::runtime_error("Only TensorND supports slicing");
			return (*ptr)[idx]; };
		T& operator()(const std::vector<size_t>& indices) {
			return (*impl)(indices); };

		std::vector<size_t> get_shape() const { 
			return impl->get_shape(); };
		std::vector<size_t> get_strides() const {
			return impl->get_strides; };
		size_t size() const {
			return impl->size(); };
		size_t ndim() const {
			return impl->get_shape().size(); };
		bool empty() const { 
			return impl->empty(); };

		void show() const {
			impl->show(); };

	};
}

