#pragma once

#include <vector>
#include <cstddef>

class Variable;

Variable reshape(const Variable& x, std::vector<size_t> shape);
Variable transpose(const Variable& x, std::vector<size_t> axes={});
Variable concat(const std::vector<Variable>& xs, int axis = 1);
Variable upsample(const Variable& x, size_t scale_factor = 2);
