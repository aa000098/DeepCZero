#include "ops/ops.hpp"
#include "function/function_all.hpp"

#include <memory>

Variable sum(const Variable &x, const std::vector<int> axis, bool keepdims) {
	using namespace function;
	std::shared_ptr<Function> f = std::make_shared<Sum>(axis, keepdims);
	return (*f)({x});
}

Variable broadcast_to(const Variable &x, const std::vector<size_t> shape) {
	using namespace function;
	if (x.shape() == shape) 
		return x;
	std::shared_ptr<Function> f = std::make_shared<Broadcast_To>(shape);
	return (*f)({x});
}

Variable sum_to(const Variable &x, const std::vector<size_t> shape) {
	using namespace function;
	if (x.shape() == shape) 
		return x;
	std::shared_ptr<Function> f = std::make_shared<Sum_To>(shape);
	return (*f)({x});
}

