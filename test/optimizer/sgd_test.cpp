#include "deepczero.hpp"

#include <iostream>
#include <cmath>

double pi = std::acos(-1.0f);

void test_sgd_update() {
	std::cout << "[TEST SGD Update]" << std::endl;

	Variable x(rand(10, 1));
	Variable y(sin(2 * pi * x) + rand(10, 1));

	float lr = 0.2;
	size_t max_iter = 10000;
	size_t hidden_size = 10;

	MLP model({hidden_size, 1});
	SGD optimizer(lr);
	optimizer.setup(model);

	Variable y_pred;
	Variable loss;

	for (size_t i = 0; i < max_iter; i++) {
		y_pred = model(x);
		loss = mean_squared_error(y, y_pred);

		model.cleargrads();
		loss.backward();

		optimizer.update();
		if (i % 1000 == 0) 
			loss.show();
	}

}

int main() {
	test_sgd_update();
}
