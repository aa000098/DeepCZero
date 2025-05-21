#include "ops/ops.hpp"
#include "function/function_all.hpp"

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
	Tensor b_tensor(a.shape(), b);
	Variable b_var(b_tensor);
	std::shared_ptr<Function> f = std::make_shared<Add>();
	return (*f)({a, b_var});
}

Variable add(const float& a, const Variable &b) {
	Tensor a_tensor(b.shape(), a);
	Variable a_var(a_tensor);
	std::shared_ptr<Function> f = std::make_shared<Add>();
	return (*f)({a_var, b});
}
Variable mul(const Variable &a, const Variable &b) {
	std::shared_ptr<Function> f = std::make_shared<Mul>();
	return (*f)({a, b});
}

Variable mul(const Variable &a, const float& b) {
	Tensor b_tensor(a.shape(), b);
	Variable b_var(b_tensor);
	std::shared_ptr<Function> f = std::make_shared<Mul>();
	return (*f)({a, b_var});
}

Variable mul(const float& a, const Variable &b) {
	Tensor a_tensor(b.shape(), a);
	Variable a_var(a_tensor);
	std::shared_ptr<Function> f = std::make_shared<Mul>();
	return (*f)({a_var, b});
}
Variable neg(const Variable &x) {
	std::shared_ptr<Function> f = std::make_shared<Neg>();
	return (*f)({x});
}

Variable sub(const Variable &a, const Variable &b) {
	std::shared_ptr<Function> f = std::make_shared<Sub>();
	return (*f)({a, b});
}
Variable sub(const Variable &a, const float &b) {
	Tensor b_tensor(a.shape(), b);
	Variable b_var(b_tensor);
	std::shared_ptr<Function> f = std::make_shared<Sub>();
	return (*f)({a, b_var});
}
Variable sub(const float &a, const Variable &b) {
	Tensor a_tensor(b.shape(), a);
	Variable a_var(a_tensor);
	std::shared_ptr<Function> f = std::make_shared<Sub>();
	return (*f)({a_var, b});
}
Variable div(const Variable &a, const Variable &b) {
	std::shared_ptr<Function> f = std::make_shared<Div>();
	return (*f)({a, b});
}
Variable div(const Variable &a, const float &b) {
	Variable b_var({b});
	std::shared_ptr<Function> f = std::make_shared<Div>();
	return (*f)({a, b_var});
}
Variable div(const float &a, const Variable &b) {
	Variable a_var({a});
	std::shared_ptr<Function> f = std::make_shared<Div>();
	return (*f)({a_var, b});
}
Variable pow(const Variable &a, const float &b) {
	Variable b_var({b});
	std::shared_ptr<Function> f = std::make_shared<Pow>();
	return (*f)({a, b_var});
}

Variable sin(const Variable &x) {
	std::shared_ptr<Function> f = std::make_shared<Sin>();
	return (*f)({x});
}
Variable cos(const Variable &x) {
	std::shared_ptr<Function> f = std::make_shared<Cos>();
	return (*f)({x});
}
Variable tanh(const Variable &x) {
	std::shared_ptr<Function> f = std::make_shared<Tanh>();
	return (*f)({x});
}

Variable matmul(const Variable &x, const Variable& w) {
	std::shared_ptr<Function> f = std::make_shared<MatMul>();
	return (*f)({x, w});
}
