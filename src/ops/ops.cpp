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

Variable add(const Variable &a, const float& b) {
	Variable b_var({b});
	std::shared_ptr<Function> f = std::make_shared<Add>();
	return (*f)({a, b_var});
}

Variable add(const float& a, const Variable &b) {
	Variable a_var({a});
	std::shared_ptr<Function> f = std::make_shared<Add>();
	return (*f)({a_var, b});
}

Variable mul(const Variable &a, const Variable &b) {
	std::shared_ptr<Function> f = std::make_shared<Mul>();
	return (*f)({a, b});
}

Variable mul(const Variable &a, const float& b) {
	Variable b_var({b});
	std::shared_ptr<Function> f = std::make_shared<Mul>();
	return (*f)({a, b_var});
}

Variable mul(const float& a, const Variable &b) {
	Variable a_var({a});
	std::shared_ptr<Function> f = std::make_shared<Mul>();
	return (*f)({a_var, b});
}

Variable neg(const Variable &x) {
	std::shared_ptr<Function> f = std::make_shared<Neg>();
	return (*f)({x});
}
/*
Variable sub(const Variable &a, const Variable &b);
Variable sub(const Variable &a, const float &b);
Variable sub(const float &a, const Variable &b);

Variable div(const Variable &a, const Variable &b);
Variable div(const Variable &a, const float &b);
Variable div(const float &a, const Variable &b);

Variable pow(const Variable &a, const float &b);
*/
