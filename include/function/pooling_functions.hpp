#pragma once

#include "function/function.hpp"
#include "container/tensor/tensor_all.hpp"

namespace function {

// [Max Pooling]
	class Pooling : public Function {
		private:
			Tensor<size_t> indexes;
			std::vector<size_t> input_shape;
			std::pair<size_t, size_t> kernel_size;
			std::pair<size_t, size_t> stride;
			std::pair<size_t, size_t> pad;

		public:
			Pooling(std::pair<size_t, size_t> kernel_size,
					std::pair<size_t, size_t> stride = {1, 1},
					std::pair<size_t, size_t> pad = {0, 0})
		: kernel_size(kernel_size),
			stride(stride),
			pad(pad) {};

			Variable forward(const std::vector<Variable>& xs) override;
			std::vector<Variable> backward(const Variable& gy) override;
			~Pooling() = default;

	};

	class Pooling2DGrad : public Function {
		private:
			Tensor<size_t> indexes;
			std::vector<size_t> input_shape;
			std::pair<size_t, size_t> kernel_size;
			std::pair<size_t, size_t> stride;
			std::pair<size_t, size_t> pad;

		public:
			Pooling2DGrad(
					Tensor<size_t> indexes,
					std::vector<size_t> input_shape,
					std::pair<size_t, size_t> kernel_size,
					std::pair<size_t, size_t> stride = {1, 1},
					std::pair<size_t, size_t> pad = {0, 0})
		: indexes(indexes),
			input_shape(input_shape),
			kernel_size(kernel_size),
			stride(stride),
			pad(pad) {};

			Variable forward(const std::vector<Variable>& xs) override;
			std::vector<Variable> backward(const Variable& gy) override;
			~Pooling2DGrad() = default;

	};

	class Pooling2DWithIndexes : public Function {
		private:
			Tensor<size_t> indexes;
			std::vector<size_t> input_shape;
			std::pair<size_t, size_t> kernel_size;
			std::pair<size_t, size_t> stride;
			std::pair<size_t, size_t> pad;

		public:
			Pooling2DWithIndexes(
					Tensor<size_t> indexes,
					std::vector<size_t> input_shape,
					std::pair<size_t, size_t> kernel_size,
					std::pair<size_t, size_t> stride = {1, 1},
					std::pair<size_t, size_t> pad = {0, 0})
		: indexes(indexes),
			input_shape(input_shape),
			kernel_size(kernel_size),
			stride(stride),
			pad(pad) {};

			Variable forward(const std::vector<Variable>& xs) override;
			std::vector<Variable> backward(const Variable& gy) override;
			~Pooling2DWithIndexes() = default;

	};


}
