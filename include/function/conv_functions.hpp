#pragma once

#include "function/function.hpp"

namespace function {

	class Im2col : public Function {
	private:
		std::vector<size_t> input_shape;
		std::pair<size_t, size_t> kernel_size;
		std::pair<size_t, size_t> stride;
		std::pair<size_t, size_t> pad;
		bool to_matrix;
	
	public:
		Im2col(	std::pair<size_t, size_t> kernel_size,
				std::pair<size_t, size_t> stride,
				std::pair<size_t, size_t> pad,
				bool to_matrix)
		: kernel_size(kernel_size),
			stride(stride),
			pad(pad),
			to_matrix(to_matrix) {};

		Variable forward(const std::vector<Variable>& xs) override;
		std::vector<Variable> backward(const Variable& gy) override;
		~Im2col() = default;

	};

}
