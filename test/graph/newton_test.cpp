#include"deepczero.hpp"

#include <iostream>
#include <cmath>

Variable f(Variable x) {
	Variable y = (x^4) - 2*(x^2);
	return y;
}
float gx2(float x) {
	return 12*pow(x, 2) - 4;
}

int main() {
	Variable x({2});

	int iters = 10;

	for (int i = 0; i < iters; i++) {
		std::cout << "[" << i << "]: " << std::endl;
		x.show();

		Variable y = f(x);

		x.cleargrad();
		y.backward();

		Tensor g = x.grad()->data();
		x = Variable(x.data() - g/gx2(x.data().raw_data()[0]));
	}
}
