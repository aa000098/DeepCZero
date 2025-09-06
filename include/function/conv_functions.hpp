#pragma once

#include "function/function.hpp"

namespace function {

// [Convolution]

	class Conv2d : public Function {
	private:
		std::pair<size_t, size_t> stride;
		std::pair<size_t, size_t> pad;

	public:
		Conv2d(	std::pair<size_t, size_t> stride = {1,1},
				std::pair<size_t, size_t> pad = {0,0})
			: stride(stride), pad(pad) {};
		
		Variable forward(const std::vector<Variable>& xs) override;
		std::vector<Variable> backward(const Variable& gy) override;
		~Conv2d() = default;

	};

	class Deconv2d : public Function {
	private:
		std::pair<size_t, size_t> stride;
		std::pair<size_t, size_t> pad;
		std::pair<size_t, size_t> out_size;
	
	public:
		Deconv2d(std::pair<size_t, size_t> stride = {1,1},
				std::pair<size_t, size_t> pad = {0,0},
				std::pair<size_t, size_t> out_size = {0, 0})
			: stride(stride), pad(pad), out_size(out_size) {};

		Variable forward(const std::vector<Variable>& xs) override;
		std::vector<Variable> backward(const Variable& gy) override;
		~Deconv2d() = default;

	};

	class Conv2dGradW : public Function {
	private:
		std::pair<size_t, size_t> kernel_size;
		std::pair<size_t, size_t> stride;
		std::pair<size_t, size_t> pad;
	public:
		Conv2dGradW(const Variable &W,
					std::pair<size_t, size_t> stride,
					std::pair<size_t, size_t> pad) {
			auto k_shape = W.shape();
			kernel_size = {k_shape[2], k_shape[3]};
			this->stride = stride;
			this->pad = pad;
		};
					
		Variable forward(const std::vector<Variable>& xs) override;
		std::vector<Variable> backward(const Variable& gy) override;
		~Conv2dGradW() = default;
	};

// [Im2col]
	class Im2col : public Function {
	private:
		std::vector<size_t> input_shape;
		std::pair<size_t, size_t> kernel_size;
		std::pair<size_t, size_t> stride;
		std::pair<size_t, size_t> pad;
		bool to_matrix;
	
	public:
		Im2col(	std::pair<size_t, size_t> kernel_size,
				std::pair<size_t, size_t> stride,
				std::pair<size_t, size_t> pad,
				bool to_matrix)
		: kernel_size(kernel_size),
			stride(stride),
			pad(pad),
			to_matrix(to_matrix) {};

		Variable forward(const std::vector<Variable>& xs) override;
		std::vector<Variable> backward(const Variable& gy) override;
		~Im2col() = default;

	};

	class Col2im : public Function {
	private:
		std::vector<size_t> input_shape;
		std::pair<size_t, size_t> kernel_size;
		std::pair<size_t, size_t> stride;
		std::pair<size_t, size_t> pad;
		bool to_matrix;
	
	public:
		Col2im(	std::vector<size_t> input_shape,
				std::pair<size_t, size_t> kernel_size,
				std::pair<size_t, size_t> stride,
				std::pair<size_t, size_t> pad,
				bool to_matrix)
		: 	input_shape(input_shape),
			kernel_size(kernel_size),
			stride(stride),
			pad(pad),
			to_matrix(to_matrix) {};

		Variable forward(const std::vector<Variable>& xs) override;
		std::vector<Variable> backward(const Variable& gy) override;
		~Col2im() = default;

	};

}
