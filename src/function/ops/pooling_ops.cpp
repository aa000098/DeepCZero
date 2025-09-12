#include "function/ops/pooling_ops.hpp"
#include "function/function_all.hpp"
#include "utils/utils.hpp"

#include <memory>
#include <cassert>

Variable pooling(const Variable& x,
				std::pair<size_t, size_t> kernel_size,
				std::pair<size_t, size_t> stride,
				std::pair<size_t, size_t> pad) {
	using namespace function;
	std::shared_ptr<Function> f = std::make_shared<Pooling>(kernel_size, stride, pad);
	return (*f)({x});
}

Variable pooling(const Variable& x,
				std::initializer_list<size_t> kernel_size,
				std::initializer_list<size_t> stride,
				std::initializer_list<size_t> pad) {
	using namespace function;
	std::shared_ptr<Function> f = std::make_shared<Pooling>(to_pair(kernel_size), to_pair(stride), to_pair(pad));
	return (*f)({x});
}
				
Variable pooling2d_grad(
		const Variable& gy,
		tensor::Tensor<size_t>& indexes,
		std::vector<size_t> input_shape,
		std::pair<size_t, size_t> kernel_size,
		std::pair<size_t, size_t> stride={1,1},
		std::pair<size_t, size_t> pad={0,0}) {
	using namespace function;
	std::shared_ptr<Function> f = std::make_shared<Pooling2DGrad>(indexes, input_shape, kernel_size, stride, pad);
	return (*f)({gy});
}

Variable pooling2d_with_indexes(
		const Variable& x,
		Tensor<size_t>& indexes,
		std::vector<size_t> input_shape,
		std::pair<size_t, size_t> kernel_size,
		std::pair<size_t, size_t> stride={1,1},
		std::pair<size_t, size_t> pad={0,0}) {
	using namespace function;
	std::shared_ptr<Function> f = std::make_shared<Pooling2DWithIndexes>(indexes, input_shape, kernel_size, stride, pad);
	return (*f)({x});
}
				
