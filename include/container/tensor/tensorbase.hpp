#pragma once

#include "container/tensor/tensorview.hpp"

#include <cstddef>
#include <vector>

namespace tensor {
	template <typename T>
	class TensorBase {
	public:

		virtual T& operator()(const std::vector<size_t>& indices) = 0;
		virtual size_t size() const = 0;
		virtual size_t ndim() const = 0;
		virtual bool empty() const = 0;
		virtual std::vector<size_t> get_shape() const = 0;
		virtual std::vector<T>& raw_data() = 0;
		virtual const std::vector<T>& raw_data() const = 0;

		virtual void show() const = 0;

		virtual ~TensorBase() = default;
	};
}
