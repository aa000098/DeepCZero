#include "deepczero.hpp"

Variable rosenbrock(Variable x0, Variable x1) {
	Variable y = (100 * ((x1 - (x0^2))^2)) + ((1 - x0)^2);
	return y;
}

int main() {
	Variable x0({0});
	Variable x1({2});

	float lr = 0.001;
	int iters = 5000;

	for (int i = 0; i < iters; i++) {
		x0.show();
		x1.show();

		Variable y = rosenbrock(x0, x1);

		x0.cleargrad();
		x1.cleargrad();
		y.backward();

		float g0 = x0.grad().raw_data()[0];
		float g1 = x1.grad().raw_data()[0];
		x0 = Variable({x0.data().raw_data()[0] - lr * g0});
		x1 = Variable({x1.data().raw_data()[0] - lr * g1});
	}
}
