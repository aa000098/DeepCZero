#pragma once

#include "function/function.hpp"

#include <stdexcept>
#include <string>

namespace function {

class Reshape: public Function {
private:
	std::vector<size_t> shape;
	std::vector<size_t> x_shape;

public:
	Reshape(const std::vector<size_t>& shape) : shape(shape) {};

public:
	Variable forward(const std::vector<Variable>& xs) override;
	std::vector<Variable> backward(const Variable& gy) override;
	~Reshape() = default;
};

class Concat : public Function {
private:
	int axis;
	std::vector<size_t> split_sizes;

public:
	Concat(int axis = 1) : axis(axis) {};
	Variable forward(const std::vector<Variable>& xs) override;
	std::vector<Variable> backward(const Variable& gy) override;
	~Concat() = default;
};

class Transpose : public Function {
private:
    std::vector<size_t> axes;

public:
    Transpose(const std::vector<size_t>& axes = {}) : axes(axes) {};

public:
	Variable forward(const std::vector<Variable>& xs) override;
	std::vector<Variable> backward(const Variable& gy) override;
	~Transpose() = default;

};

class Upsample : public Function {
private:
	size_t scale_factor;
	bool bilinear;
	bool align_corners;
	std::vector<size_t> input_shape;

public:
	Upsample(size_t scale_factor = 2, const std::string& mode = "nearest", bool align_corners = false)
		: scale_factor(scale_factor), bilinear(mode == "bilinear"), align_corners(align_corners) {
		if (mode != "nearest" && mode != "bilinear")
			throw std::invalid_argument("Upsample: mode must be \"nearest\" or \"bilinear\"");
	};
	Variable forward(const std::vector<Variable>& xs) override;
	std::vector<Variable> backward(const Variable& gy) override;
	~Upsample() = default;
};

class Gather : public Function {
private:
	std::vector<size_t> indices;
	size_t src_rows;

public:
	Gather(const std::vector<size_t>& indices) : indices(indices) {};
	Variable forward(const std::vector<Variable>& xs) override;
	std::vector<Variable> backward(const Variable& gy) override;
	~Gather() = default;
};

}
