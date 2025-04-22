#pragma once

#include <cstddef>
#include <vector>

namespace tensor {
	template <typename T>
	class TensorBase {
	public:
		virtual size_t size() const = 0;
		virtual size_t ndim() const = 0;
		virtual bool empty() const = 0;
		virtual std::vector<size_t> shape() const = 0;

		virtual ~TensorBase() = default;
	};
}
