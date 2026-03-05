#pragma once

#include "function/function.hpp"

namespace function {

class RMSNormFunc : public Function {
private:
	float eps;

public:
	RMSNormFunc(float eps = 1e-5f) : eps(eps) {}

	// inputs: [x, weight]
	// x: [batch, seq_len, hidden_size]
	// weight: [hidden_size]
	Variable forward(const std::vector<Variable>& xs) override;
	std::vector<Variable> backward(const Variable& gy) override;

	~RMSNormFunc() = default;
};

}
