#include "deepczero.hpp"

#include <cmath>

double pi = std::acos(-1.0f);

Variable predict(Variable x, 
				Variable W1, 
				Variable b1, 
				Variable W2, 
				Variable b2) {
	Variable y = linear(x, W1, b1);
	y = sigmoid(y);
	y = linear(y, W2, b2);
	return y;

}

void neural_network_test() {
    std::cout << "▶️ Running Neural Network test...\n";

	Variable x = rand(	/* rows */	10, 
						/* cols */	1, 
						/* seed */	0);
 	
	Variable y = sin(2 * pi * x) + rand(10, 1);

	size_t I = 1;
	size_t H = 10;
	size_t O = 1;

	Variable W1 = Variable(0.01 * randn(I, H)); 
	Variable b1(H);
	Variable W2 = Variable(0.01 * randn(H, O)); 
	Variable b2(O);
	W1.set_name("W1");
	b1.set_name("b1");
	W2.set_name("W2");
	b2.set_name("b2");

	float lr = 0.2;
	size_t iters = 10000;

	for (size_t i = 0; i < iters; i++) {
		Variable y_pred = predict(x, W1, b1, W2, b2);
		y_pred.show();
		Variable loss = mean_squared_error(y, y_pred);
		loss.set_name("loss");
		loss.show();

		W1.cleargrad();
		b1.cleargrad();
		W2.cleargrad();
		b2.cleargrad();
		loss.backward();


		W1.show();
		b1.show();
		W2.show();
		b2.show();
		W1.data() -= lr * W1.grad().data();
		b1.data() -= lr * b1.grad().data();
		W2.data() -= lr * W2.grad().data();
		b2.data() -= lr * b2.grad().data();
		W1.show();

		if (i % 1000 == 0) loss.show();

	}
	
}

int main() {
	neural_network_test();
	return 0;
}
