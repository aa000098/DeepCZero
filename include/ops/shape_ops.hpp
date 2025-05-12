#pragma once

#include <vector>
#include <cstddef>

class Variable;

Variable reshape(const Variable& x, std::vector<size_t> shape);
Variable transpose(const Variable& x, std::vector<size_t> axes={});
