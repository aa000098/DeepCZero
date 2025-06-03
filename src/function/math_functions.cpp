#include "function/math_functions.hpp"
#include "container/tensor/tensor_all.hpp"

Variable function::Square::forward(const std::vector<Variable>& xs) {
	const Tensor<>& x_data = xs[0].data();
	Tensor<> result = x_data * x_data;
	return Variable(result);
}

std::vector<Variable> function::Square::backward(const Variable& gy) {
	const Variable& x = inputs[0];
	return { 2.0f * x * gy};
}

Variable function::Add::forward(const std::vector<Variable>& xs) {
	const Tensor<>& a = xs[0].data();
	const Tensor<>& b = xs[1].data();
	
	x0_shape = a.get_shape();
	x1_shape = b.get_shape();

	Tensor<> result = a + b;
	return Variable(result);
}

std::vector<Variable> function::Add::backward(const Variable& gy) {
	Variable gx0 = gy;
	Variable gx1 = gy;

	if (x0_shape != x1_shape) {
		gx0 = sum_to(gx0, x0_shape);
		gx1 = sum_to(gx1, x1_shape);
	}
	return {gx0, gx1};
}

Variable function::Mul::forward(const std::vector<Variable>& xs) {
	const Tensor<>& a = xs[0].data();
	const Tensor<>& b = xs[1].data();

	x0_shape = a.get_shape();
	x1_shape = b.get_shape();
	
	Tensor<> result = a * b;
	return Variable(result);
}

std::vector<Variable> function::Mul::backward(const Variable& gy) {
	const Variable& a = inputs[0];
	const Variable& b = inputs[1];

	Variable gx0 = b*gy;
	Variable gx1 = a*gy;
	if (x0_shape != x1_shape) {
		gx0 = sum_to(gx0, x0_shape);
		gx1 = sum_to(gx1, x1_shape);
	}
	return {gx0, gx1};
}

Variable function::Neg::forward(const std::vector<Variable>& xs) {
	const Tensor<>& x = xs[0].data();
	Tensor<> result = -x;
	return Variable(result);
}

std::vector<Variable> function::Neg::backward(const Variable& gy) {
	return {-gy};
}

Variable function::Sub::forward(const std::vector<Variable>& xs) {
	const Tensor<>& a = xs[0].data();
	const Tensor<>& b = xs[1].data();

	x0_shape = a.get_shape();
	x1_shape = b.get_shape();

	Tensor<> result = a - b;
	return Variable(result);
}

std::vector<Variable> function::Sub::backward(const Variable& gy) {
	Variable gx0 = gy;
	Variable gx1 = -gy;

	if (x0_shape != x1_shape) {
		gx0 = sum_to(gx0, x0_shape);
		gx1 = sum_to(gx1, x1_shape);
	}
	return {gx0, gx1};
}

Variable function::Div::forward(const std::vector<Variable>& xs) {
	const Tensor<>& a = xs[0].data();
	const Tensor<>& b = xs[1].data();

	x0_shape = a.get_shape();
	x1_shape = b.get_shape();

	Tensor result = a / b;
	return Variable(result);
}

std::vector<Variable> function::Div::backward(const Variable& gy) {
	const Variable& a = inputs[0];
	const Variable& b = inputs[1];
 
	Variable gx0 = gy / b;
	Variable gx1 = -gy * a / (b * b);
	if (x0_shape != x1_shape) {
		gx0 = sum_to(gx0, x0_shape);
		gx1 = sum_to(gx1, x1_shape);
	}
	return {gx0, gx1};
}

Variable function::Pow::forward(const std::vector<Variable>& xs) {
	const Tensor<>& base = xs[0].data();
	float scalar = xs[1].data().raw_data()[0];

	Tensor result = pow(base, scalar);
	return Variable(result);
}

std::vector<Variable> function::Pow::backward(const Variable& gy) {
	const Variable& a = inputs[0];
	const Variable& b = inputs[1];
    float scalar = b.data().raw_data()[0];

	Variable result = gy * scalar * (a^(scalar-1));
    return {result};
}

Variable function::Exp::forward(const std::vector<Variable>& xs) {
    const Tensor<>& x = xs[0].data();
    Tensor result = exp(x);
    return Variable(result);
}

std::vector<Variable> function::Exp::backward(const Variable& gy) {
	const Variable& x = inputs[0];
	return {gy * exp(x)};
}

Variable function::Sin::forward(const std::vector<Variable>& xs) {
    const Tensor<>& x = xs[0].data();
    Tensor result = sin(x);
    return Variable(result);
}

std::vector<Variable> function::Sin::backward(const Variable& gy) {
	const Variable& x = inputs[0];
	return {gy * cos(x)};
}

Variable function::Cos::forward(const std::vector<Variable>& xs) {
    const Tensor<>& x = xs[0].data();
    Tensor result = cos(x);
    return Variable(result);
}

std::vector<Variable> function::Cos::backward(const Variable& gy) {
	const Variable& x = inputs[0];
	return {gy * -sin(x)};
}

Variable function::Tanh::forward(const std::vector<Variable>& xs) {
    const Tensor<>& x = xs[0].data();
    Tensor result = tanh(x);
    return Variable(result);
}

std::vector<Variable> function::Tanh::backward(const Variable& gy) {
	const Variable& y = output.lock();
	if (y.empty()) 
		throw std::runtime_error("Tanh::backweard(): output expired");
	return {gy * (1 - y * y)};
}

Variable function::MatMul::forward(const std::vector<Variable>& xs) {
	const Tensor<>& x = xs[0].data();
	const Tensor<>& w = xs[1].data();

    Tensor result = dot(x, w);
    return Variable(result);
}

std::vector<Variable> function::MatMul::backward(const Variable& gy) {
	const Variable& x = inputs[0];
	const Variable& w = inputs[1];

	const Variable& gx = matmul(gy, w.trans());
	const Variable& gw = matmul(x.trans(), gy);

	return {gx, gw};
}

Variable function::Linear::forward(const std::vector<Variable>& xs) {
	const Tensor<>& x = xs[0].data();
	const Tensor<>& w = xs[1].data();
	const Tensor<>& b = xs[2].data();

    Tensor y = dot(x, w);
	if (!b.empty())
		y += b;
    return Variable(y);
}

std::vector<Variable> function::Linear::backward(const Variable& gy) {
	const Variable x = inputs[0];
	const Variable w = inputs[1];
	const Variable b = inputs[2];

	const Variable gx = matmul(gy, w.trans());
	const Variable gw = matmul(x.trans(), gy);
	Variable gb;
	if (!b.empty())
		gb = sum_to(gy, b.shape());
	else
		gb = Variable();

	return {gx, gw, gb}; 
}
