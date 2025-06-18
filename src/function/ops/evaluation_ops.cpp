#include "function/ops/evaluation_ops.hpp"
#include "function/evaluation_functions.hpp"

Variable accuracy(const Variable &y, const Variable& t) {
	using namespace function;
	Accuracy f;
	return f(y, t);
}
