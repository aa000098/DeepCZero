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

Variable predict(Variable x, Variable W, Variable b) {
	return matmul(x, W) + b;
}

int main() {
	Tensor x_rand = rand_tensor(100, 1);
	Tensor y_rand = 5.0f + 2.0f * x_rand + rand_tensor(100, 1);

	Variable x(x_rand);
	Variable y(y_rand);

	x.show();
	y.show();

	Variable W({0});
	Variable b({0});

}
