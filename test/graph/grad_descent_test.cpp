#include "deepczero.hpp"

Variable rosenbrock(Variable x0, Variable x1) {
	Variable y = (100 * ((x1 - (x0^2))^2)) + ((1 - x0)^2);
	return y;
}

int main() {
	Variable x0({0});
	Variable x1({2});

	float lr = 0.001;
	int iters = 50000;

	for (int i = 0; i < iters; i++) {
		Variable y = rosenbrock(x0, x1);

		x0.cleargrad();
		x1.cleargrad();
		y.backward();

		Tensor g0 = x0.grad().data();
		Tensor g1 = x1.grad().data();
		x0 = Variable(x0.data() - (lr * g0));
		x1 = Variable(x1.data() - (lr * g1));
	}
	x0.show();
	x1.show();
}
