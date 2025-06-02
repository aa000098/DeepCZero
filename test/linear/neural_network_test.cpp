#include "deepczero.hpp"

#include <cmath>
#include <fstream>

double pi = std::acos(-1.0f);

Variable predict(Variable &x, 
				Variable &W1, 
				Variable &b1, 
				Variable &W2, 
				Variable &b2) {
	Variable y = linear(x, W1, b1);
	y = sigmoid(y);
	y = linear(y, W2, b2);
	return y;

}

void neural_network_test() {
    std::cout << "▶️ Running Neural Network test...\n";

	// const size_t N = 100;
	const size_t N = 10;
	Variable x = rand(	/* rows */	N, 
						/* cols */	1, 
						/* seed */	0);
	x.set_name("x");
 	
	Variable y = sin(2 * pi * x) + rand(N, 1);
	y.set_name("y");

	size_t I = 1;
	size_t H = 10;
	size_t O = 1;

	Variable W1 = Variable(0.01f * randn(I, H)); 
	Variable b1(H);
	Variable W2 = Variable(0.01f * randn(H, O)); 
	Variable b2(O);
	W1.set_name("W1");
	b1.set_name("b1");
	W2.set_name("W2");
	b2.set_name("b2");

	float lr = 0.2;
	size_t iters = 10000;

	Variable y_pred;
	Variable loss;
	
	for (size_t i = 0; i < iters; i++) {
		y_pred = predict(x, W1, b1, W2, b2);
		y_pred.set_name("y_pred");
		loss = mean_squared_error(y, y_pred);
		loss.set_name("loss");

		W1.cleargrad();
		b1.cleargrad();
		W2.cleargrad();
		b2.cleargrad();
		loss.backward();

		W1.data() -= lr * W1.grad().data();
		b1.data() -= lr * b1.grad().data();
		W2.data() -= lr * W2.grad().data();
		b2.data() -= lr * b2.grad().data();

		if (i % 1000 == 0) loss.show();

	}
/*	
	std::ofstream out("/home/user/.deepczero/data/linear_output.csv");
	for (size_t i = 0; i < x.shape()[0]; ++i) {
    	out << x.data().raw_data()[i] << "," << y.data().raw_data()[i] << "," << y_pred.data().raw_data()[i] << "\n";
	}
	out.close();
*/	
}

int main() {
	neural_network_test();
	return 0;
}
