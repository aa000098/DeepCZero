#include "function/ops/ops.hpp"
#include "function/function_all.hpp"

#include <memory>

Variable reshape(const Variable &x, const std::vector<size_t> shape) {
	using namespace function;
	if (x.shape() == shape) 
		return x;
	std::shared_ptr<Function> f = std::make_shared<Reshape>(shape);
	return (*f)({x});
}

Variable transpose(const Variable &x, const std::vector<size_t> axes) {
	using namespace function;
	std::shared_ptr<Function> f = std::make_shared<Transpose>(axes);
	return (*f)({x});
}
