#include "deepczero.hpp"
#include <random>


Tensor<> rand_tensor(size_t rows, size_t cols, size_t seed = 42) {
	//std::random_device rd;
	//std::mt19937 gen(rd());
	std::mt19937 gen(seed);

	std::uniform_real_distribution<> dist(-1, 1);

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
	return sum(diff^2) / diff.size();
}

int main() {
	Tensor x_rand = rand_tensor(100, 1);
	Tensor y_rand = 5.0f + 2.0f * x_rand + rand_tensor(100, 1, 23);

//	Tensor x_rand({3,1}, {1,2,3});
//	Tensor y_rand = 5.0f + 2.0f * x_rand + Tensor({3,1}, {1,2,3});

	Variable x(x_rand);
	Variable y(y_rand);

	Tensor w_tensor({1,1});
	Tensor b_tensor({1});
	Variable W(w_tensor);
	Variable b(b_tensor);

	W.set_name("W");
	b.set_name("b");

	float lr = 0.1;
	size_t iters = 100;

	Variable loss;
	

	for (size_t i = 0; i < iters; i++) {
		Variable y_pred = predict(x, W, b);
		loss = mean_squared_error(y, y_pred);

		W.cleargrad();
		b.cleargrad();
		loss.backward();

		W.data() -= lr * W.grad().data();
		b.data() -= lr * b.grad().data();
		W.set_name("W");
		b.set_name("b");
		loss.set_name("loss");
	
		W.show();
		b.show();
		loss.show();
	}
}
