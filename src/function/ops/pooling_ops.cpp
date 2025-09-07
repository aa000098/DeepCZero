#include "function/ops/pooling_ops.hpp"
#include "function/function_all.hpp"

#include <memory>

Variable pooling(const Variable& x,
				std::pair<size_t, size_t> kernel_size,
				std::pair<size_t, size_t> stride={1,1},
				std::pair<size_t, size_t> pad={0,0}) {
	using namespace function;
	std::shared_ptr<Function> f = std::make_shared<Pooling>(kernel_size, stride, pad);
	return (*f)({x});
}
				
