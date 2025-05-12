#pragma once

#include <vector>
#include <cstddef>

class Variable;

Variable sum(const Variable& x, std::vector<size_t> shape);
Variable sum_to(const Variable& x, std::vector<size_t> axes={});
