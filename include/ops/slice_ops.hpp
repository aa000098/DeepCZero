#pragma once

#include <vector>
#include <cstddef>

class Variable;

Variable get_item(const Variable& x, 
					const std::vector<size_t> slices);

Variable get_item_grad(const Variable& gy, 
						const std::vector<size_t> slices,
						const std::vector<size_t> in_shape);

