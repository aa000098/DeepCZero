#pragma once

#include <vector>
#include <cstddef>

class Variable;

Variable pooling(const Variable& x,
				std::pair<size_t, size_t> kernel_size,
				std::pair<size_t, size_t> stride,
				std::pair<size_t, size_t> pad);

