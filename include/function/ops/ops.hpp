#pragma once

#include <vector>
#include <cstddef>

class Variable;

// arithmetics
Variable square(const Variable &x);

Variable exp(const Variable &x);

Variable add(const Variable &a, const Variable &b);
Variable add(const Variable &a, const float &b);
Variable add(const float &a, const Variable &b);

Variable mul(const Variable &a, const Variable &b);
Variable mul(const Variable &a, const float &b);
Variable mul(const float &a, const Variable &b);

Variable neg(const Variable &x);

Variable sub(const Variable &a, const Variable &b);
Variable sub(const Variable &a, const float &b);
Variable sub(const float &a, const Variable &b);

Variable div(const Variable &a, const Variable &b);
Variable div(const Variable &a, const float &b);
Variable div(const float &a, const Variable &b);

Variable pow(const Variable &a, const float &b);

Variable sin(const Variable &x);
Variable cos(const Variable &x);
Variable tanh(const Variable& x);

Variable matmul(const Variable& x, const Variable& w);

// loss
Variable mean_squared_error(const Variable& x0, const Variable& x1);
Variable softmax_cross_entropy_error(const Variable& x, const Variable& t);

// layers
Variable linear(const Variable& x, const Variable& w, const Variable& b);

// activations
Variable sigmoid(const Variable& x);
Variable softmax(const Variable& x, const std::vector<int> axes = {1});
Variable relu(const Variable& x);
Variable dropout(const Variable& x, const float& dropout_rate);
