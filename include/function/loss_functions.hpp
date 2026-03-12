#pragma once

#include "function/function.hpp"

namespace function {

	class MeanSquaredError : public Function {

	public:
		Variable forward(const std::vector<Variable>& xs) override;
		std::vector<Variable> backward(const Variable& gy) override;
		~MeanSquaredError() = default;
	};

	class SoftmaxCrossEntropyError : public Function {
	public:
		Variable forward(const std::vector<Variable>& xs) override;
		std::vector<Variable> backward(const Variable& gy) override;
		~SoftmaxCrossEntropyError() = default;
	};

	class BinaryCrossEntropy : public Function {
	private:
		float pos_weight;
	public:
		BinaryCrossEntropy(float pos_weight = 1.0f) : pos_weight(pos_weight) {};
		Variable forward(const std::vector<Variable>& xs) override;
		std::vector<Variable> backward(const Variable& gy) override;
		~BinaryCrossEntropy() = default;
	};

	class Abs : public Function {
	public:
		Variable forward(const std::vector<Variable>& xs) override;
		std::vector<Variable> backward(const Variable& gy) override;
		~Abs() = default;
	};

	class Clamp : public Function {
	private:
		float min_val, max_val;
	public:
		Clamp(float min_val, float max_val) : min_val(min_val), max_val(max_val) {};
		Variable forward(const std::vector<Variable>& xs) override;
		std::vector<Variable> backward(const Variable& gy) override;
		~Clamp() = default;
	};

	// CIoU loss: Complete IoU for bounding box regression
	// inputs: pred [N, 4], target [N, 4] in (cx, cy, w, h) format
	// output: scalar loss = mean(1 - CIoU)
	class CIoU : public Function {
	public:
		Variable forward(const std::vector<Variable>& xs) override;
		std::vector<Variable> backward(const Variable& gy) override;
		~CIoU() = default;
	};

}
