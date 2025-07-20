#pragma once

#include <vector>
#include <cstddef>

class Variable;

Variable conv2d(const Variable& x,
				const Variable& W,
				const Variable& b,
				std::pair<size_t, size_t> stride,
				std::pair<size_t, size_t> pad);

Variable deconv2d(const Variable& x,
				const Variable& W,
				const Variable& b,
				std::pair<size_t, size_t> stride,
				std::pair<size_t, size_t> pad,
				std::pair<size_t, size_t> out_size);



Variable im2col(const Variable& x,
				std::pair<size_t, size_t> kernel_size,
				std::pair<size_t, size_t> stride,
				std::pair<size_t, size_t> pad,
				bool to_matrix);

Variable col2im(const Variable& x,
				std::vector<size_t> input_shape,
				std::pair<size_t, size_t> kernel_size,
				std::pair<size_t, size_t> stride,
				std::pair<size_t, size_t> pad,
				bool to_matrix);
