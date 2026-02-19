#pragma once

#include "function/function.hpp"
#include "container/tensor/tensor_all.hpp"

namespace function {

class BatchNorm2dFunc : public Function {
private:
	float eps;
	bool is_training;
	Tensor<> saved_mean;
	Tensor<> saved_inv_std;
	size_t N, C, H, W;

public:
	BatchNorm2dFunc(float eps = 1e-5, bool is_training = true)
		: eps(eps), is_training(is_training) {};

	// inputs: [x, gamma, beta, running_mean, running_var]
	Variable forward(const std::vector<Variable>& xs) override;
	std::vector<Variable> backward(const Variable& gy) override;

	const Tensor<>& get_saved_mean() const { return saved_mean; }
	const Tensor<>& get_saved_inv_std() const { return saved_inv_std; }

	~BatchNorm2dFunc() = default;
};

}
