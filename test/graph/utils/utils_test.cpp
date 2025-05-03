#include "deepczero.hpp"

#include <random>

Tensor<> random_tensor(
		const std::vector<size_t>& shape, 
		float low = -1.0f, 
		float high = 1.0f) {
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<float> dist(low, high);
	
	size_t total_size = 1;
	for (size_t dim : shape)
		total_size *= dim;

	std::vector<float> data(total_size);
	for (auto& v : data)
		v = dist(mt);

	return Tensor(shape, data);
}

Variable random_variable(
		const std::vector<size_t>& shape, 
		const std::string& name = "") {
	return Variable(random_tensor(shape), name);
}

int main() {
	Variable x = random_variable({2,3}, "x");
	std::cout << _dot_var(x) << std::endl;
	std::cout << _dot_var(x, true) << std::endl;
	x.show();
}
