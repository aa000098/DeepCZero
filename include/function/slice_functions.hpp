#pragma once

#include "function/function.hpp"

namespace function {

	class GetItem : public Function {
	private:
		std::vector<size_t> slices;

	public:
		GetItem(const std::vector<size_t> slices) : slices(slices) {};
		GetItem(const std::initializer_list<size_t> slices) : GetItem(std::vector<size_t>(slices)) {};
		GetItem(size_t slices) : GetItem(std::vector<size_t>(slices)) {};

	public:
		Variable forward(const std::vector<Variable>& xs) override;
		std::vector<Variable> backward(const Variable& gy) override;
		~GetItem() = default;
	};


	// Slice along a specific axis: x.slice(axis, start, end)
	class SliceAxis : public Function {
	private:
		int axis;
		size_t start, end;
		std::vector<size_t> input_shape;

	public:
		SliceAxis(int axis, size_t start, size_t end)
			: axis(axis), start(start), end(end) {};
		Variable forward(const std::vector<Variable>& xs) override;
		std::vector<Variable> backward(const Variable& gy) override;
		~SliceAxis() = default;
	};

	class GetItemGrad : public Function {
	private:
		std::vector<size_t> slices;
		std::vector<size_t> in_shape;

	public:
		GetItemGrad(const std::vector<size_t> slices,
					const std::vector<size_t> in_shape)
			: slices(slices), in_shape(in_shape) {};

	public:
		Variable forward(const std::vector<Variable>& xs) override;
		std::vector<Variable> backward(const Variable& gy) override;
		~GetItemGrad() = default;

	};
}
