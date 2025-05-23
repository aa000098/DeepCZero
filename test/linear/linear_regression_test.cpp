#include "deepczero.hpp"
#include <random>


Tensor<> rand_tensor(size_t rows, size_t cols) {
	//std::random_device rd;
	//std::mt19937 gen(rd());
	std::mt19937 gen(42);

	std::uniform_real_distribution<> dist(0, 10);

	std::vector<float> data;
	data.reserve(rows * cols);
	for (size_t i = 0; i < rows * cols; i++)
		data.push_back(dist(gen));

	return Tensor<>({rows, cols}, data);
}

Variable predict(Variable& x, Variable& W, Variable& b) {
	return matmul(x, W) + b;
}

Variable mean_squared_error(Variable& x0, Variable& x1) {
	Variable diff = x0 - x1;
	sum(diff^2).show();
	return sum(diff^2) / diff.size();
}

int main() {
	//Tensor x_rand = rand_tensor(100, 1);
	//Tensor y_rand = 5.0f + 2.0f * x_rand + rand_tensor(100, 1);

	Tensor x_rand({3,1}, {1,2,3});
	Tensor y_rand = 5.0f + 2.0f * x_rand + Tensor({3,1}, {1,2,3});

	Variable x(x_rand);
	Variable y(y_rand);

	x.show();
	y.show();

	Tensor w_tensor({1,1});
	Tensor b_tensor({1});
	Variable W(w_tensor);
	Variable b(b_tensor);

	W.set_name("W");
	b.set_name("b");

	float lr = 0.1;
	size_t iters = 100;

	Variable loss;
	W.show();
	b.show();

	for (size_t i = 0; i < iters; i++) {
		Variable y_pred = predict(x, W, b);
		loss = mean_squared_error(y, y_pred);
		y_pred.set_name("y_pred");	
		W.set_name("W1");
		b.set_name("b1");
		loss.set_name("loss1");

		y_pred.show();
		W.show();
		b.show();
		loss.show();

		W.cleargrad();
		b.cleargrad();
		loss.backward();
		W.set_name("W2");
		b.set_name("b2");
		loss.set_name("loss2");

		W.show();
		b.show();

		W.data() -= lr * W.grad().data();
		b.data() -= lr * b.grad().data();
		W.set_name("W3");
		b.set_name("b3");
		loss.set_name("loss3");
	
		W.show();
		b.show();
		loss.show();
	}
}
