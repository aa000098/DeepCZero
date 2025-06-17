#include "function/ops/ops.hpp"
#include "function/function_all.hpp"

#include <memory>


Variable square(const Variable &x) {
	using namespace function;
	std::shared_ptr<Function> f = std::make_shared<Square>();
	return (*f)(x);
}

Variable exp(const Variable &x) {
	using namespace function;
	std::shared_ptr<Function> f = std::make_shared<Exp>();
	return (*f)({x});
}

Variable add(const Variable &a, const Variable &b) {
	using namespace function;
	std::shared_ptr<Function> f = std::make_shared<Add>();
	return (*f)({a, b});
}
Variable add(const Variable &a, const float& b) {
	using namespace function;
	Tensor b_tensor(a.shape(), b);
	Variable b_var(b_tensor);
	std::shared_ptr<Function> f = std::make_shared<Add>();
	return (*f)({a, b_var});
}

Variable add(const float& a, const Variable &b) {
	using namespace function;
	Tensor a_tensor(b.shape(), a);
	Variable a_var(a_tensor);
	std::shared_ptr<Function> f = std::make_shared<Add>();
	return (*f)({a_var, b});
}
Variable mul(const Variable &a, const Variable &b) {
	using namespace function;
	std::shared_ptr<Function> f = std::make_shared<Mul>();
	return (*f)({a, b});
}

Variable mul(const Variable &a, const float& b) {
	using namespace function;
	Tensor b_tensor(a.shape(), b);
	Variable b_var(b_tensor);
	std::shared_ptr<Function> f = std::make_shared<Mul>();
	return (*f)({a, b_var});
}

Variable mul(const float& a, const Variable &b) {
	using namespace function;
	Tensor a_tensor(b.shape(), a);
	Variable a_var(a_tensor);
	std::shared_ptr<Function> f = std::make_shared<Mul>();
	return (*f)({a_var, b});
}
Variable neg(const Variable &x) {
	using namespace function;
	std::shared_ptr<Function> f = std::make_shared<Neg>();
	return (*f)({x});
}

Variable sub(const Variable &a, const Variable &b) {
	using namespace function;
	std::shared_ptr<Function> f = std::make_shared<Sub>();
	return (*f)({a, b});
}
Variable sub(const Variable &a, const float &b) {
	using namespace function;
	Tensor b_tensor(a.shape(), b);
	Variable b_var(b_tensor);
	std::shared_ptr<Function> f = std::make_shared<Sub>();
	return (*f)({a, b_var});
}
Variable sub(const float &a, const Variable &b) {
	using namespace function;
	Tensor a_tensor(b.shape(), a);
	Variable a_var(a_tensor);
	std::shared_ptr<Function> f = std::make_shared<Sub>();
	return (*f)({a_var, b});
}
Variable div(const Variable &a, const Variable &b) {
	using namespace function;
	std::shared_ptr<Function> f = std::make_shared<Div>();
	return (*f)({a, b});
}
Variable div(const Variable &a, const float &b) {
	using namespace function;
	Variable b_var({b});
	std::shared_ptr<Function> f = std::make_shared<Div>();
	return (*f)({a, b_var});
}
Variable div(const float &a, const Variable &b) {
	using namespace function;
	Variable a_var({a});
	std::shared_ptr<Function> f = std::make_shared<Div>();
	return (*f)({a_var, b});
}
Variable pow(const Variable &a, const float &b) {
	using namespace function;
	Variable b_var({b});
	std::shared_ptr<Function> f = std::make_shared<Pow>();
	return (*f)({a, b_var});
}

Variable sin(const Variable &x) {
	using namespace function;
	std::shared_ptr<Function> f = std::make_shared<Sin>();
	return (*f)({x});
}
Variable cos(const Variable &x) {
	using namespace function;
	std::shared_ptr<Function> f = std::make_shared<Cos>();
	return (*f)({x});
}
Variable tanh(const Variable &x) {
	using namespace function;
	std::shared_ptr<Function> f = std::make_shared<Tanh>();
	return (*f)({x});
}

Variable matmul(const Variable &x, const Variable& w) {
	using namespace function;
	std::shared_ptr<Function> f = std::make_shared<MatMul>();
	return (*f)({x, w});
}

// loss
Variable mean_squared_error(const Variable &x0, const Variable& x1) {
	using namespace function;
	std::shared_ptr<Function> f = std::make_shared<MeanSquaredError>();
	return (*f)({x0, x1});
}

Variable softmax_cross_entropy_error(const Variable& x, const Variable& t) {
	using namespace function;
	std::shared_ptr<Function> f = std::make_shared<SoftmaxCrossEntropyError>();
	return (*f)({x, t});

}

// layer
Variable linear(const Variable& x, const Variable& w, const Variable& b) {
	using namespace function;
	std::shared_ptr<Function> f = std::make_shared<Linear>();
	return (*f)({x, w, b});
}

// activation
Variable sigmoid(const Variable& x) {
	using namespace function;
	std::shared_ptr<Function> f = std::make_shared<Sigmoid>();
	return (*f)({x});
}

Variable softmax(const Variable& x, std::vector<int> axes) {
	using namespace function;
	std::shared_ptr<Function> f = std::make_shared<Softmax>(axes);
	return (*f)({x});
}
