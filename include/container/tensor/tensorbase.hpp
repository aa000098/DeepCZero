#pragma once


#include <cstddef>
#include <vector>
#include <memory>

namespace tensor {
	template <typename T>
	class TensorBase {
	public:

		// slice functions
		virtual T& operator()(const std::vector<size_t>& indices) = 0;
		virtual std::shared_ptr<TensorBase<T>> operator[](size_t index) = 0;
		virtual std::shared_ptr<TensorBase<T>> slice(size_t dim, size_t start, size_t end) const = 0; 
		virtual std::shared_ptr<TensorBase<T>> gather_rows(const std::vector<size_t>& indices) const;

		// base functions
		virtual size_t size() const = 0;
		virtual size_t ndim() const = 0;
		virtual size_t get_offset() const = 0;
		virtual bool empty() const = 0;
		virtual std::vector<size_t> get_shape() const = 0;
		virtual std::vector<size_t> get_strides() const = 0;
		virtual std::vector<T>& raw_data() = 0;
		virtual const std::vector<T>& raw_data() const = 0;
		virtual std::shared_ptr<std::vector<T>> shared_data() const = 0;

		// io functions
		virtual void show() const = 0;

		virtual ~TensorBase() = default;
	};
}
