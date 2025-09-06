#pragma once

#include "function/function.hpp"

namespace function {

// [Max Pooling]
	class Pooling : public Function {
		private:
			std::pair<size_t, size_t> kernel_size;
			std::pair<size_t, size_t> stride;
			std::pair<size_t, size_t> pad;

		public:
			Pooling(std::pair<size_t, size_t> kernel_size,
					std::pair<size_t, size_t> stride = {1, 1},
					std::pair<size_t, size_t> pad = {0, 0})
		: kernel_size(kernel_size),
			stride(stride),
			pad(pad) {};

			Variable forward(const std::vector<Variable>& xs) override;
			std::vector<Variable> backward(const Variable& gy) override;
			~Pooling() = default;

	};

}
