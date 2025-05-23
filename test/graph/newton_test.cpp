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
// manual differential
	Variable x({2});
	x.set_name("x");
	int iters = 10;

	for (int i = 0; i < iters; i++) {
		std::cout << "[" << i << "]: " << std::endl;
		x.show();

		Variable y = f(x);
		y.set_name("y");
		x.cleargrad();
		y.backward();

		Tensor g = x.grad().data();
		x = Variable(x.data() - g/gx2(x.data().raw_data()[0]));
	}

// auto differential
	Variable x2({2});
	x2.set_name("x2");

	for (int i = 0; i < iters; i++) {
		std::cout << "[" << i << "]: " << std::endl;
		x2.show();

		Variable y2 = f(x2);
		y2.set_name("y2");
		y2.backward(true, true);

		Variable gx = x2.grad();
		gx.set_name("gx");
		x2.cleargrad();
		gx.backward();
		Variable gx2 = x2.grad();
		x2 = Variable(x2.data() - (gx.data()/gx2.data()));
	}
}
