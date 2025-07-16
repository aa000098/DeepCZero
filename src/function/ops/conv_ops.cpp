#include "function/ops/conv_ops.hpp"
#include "function/function_all.hpp"

#include <memory>

Variable im2col(const Variable& x,
				std::pair<size_t, size_t> kernel_size,
				std::pair<size_t, size_t> stride,
				std::pair<size_t, size_t> pad,
				bool to_matrix) {
	using namespace function;
	std::shared_ptr<Function> f = std::make_shared<Im2col>(kernel_size, stride, pad, to_matrix);
	return (*f)(x);
}

Variable col2im(const Variable& x,
				std::vector<size_t> input_shape,
				std::pair<size_t, size_t> kernel_size,
				std::pair<size_t, size_t> stride,
				std::pair<size_t, size_t> pad,
				bool to_matrix) {
	using namespace function;
	std::shared_ptr<Function> f = std::make_shared<Col2im>(input_shape, kernel_size, stride, pad, to_matrix);
	return (*f)(x);
}

