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
