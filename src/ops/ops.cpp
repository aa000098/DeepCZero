#include "ops/ops.hpp"
#include "function/function.hpp"

#include <memory>

Variable square(const Variable &x) {
	std::shared_ptr<Function> f = std::make_shared<Square>();
	return (*f)({x});
}

Variable exp(const Variable &x) {
	std::shared_ptr<Function> f = std::make_shared<Exp>();
	return (*f)({x});
}

Variable add(const Variable &a, const Variable &b) {
	std::shared_ptr<Function> f = std::make_shared<Add>();
	return (*f)({a, b});
}

Variable mul(const Variable &a, const Variable &b) {
	std::shared_ptr<Function> f = std::make_shared<Mul>();
	return (*f)({a, b});
}

