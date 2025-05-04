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

Variable goldstein(Variable x, Variable y) {
	Variable z = (1 + ((x+y+1)^2) * (19 - (14*x) + (3*(x^2)) - (14*y) + (6*x*y) + 3*(y^2))) * (30 + ((2*x - 3*y)^2) * (18 - 32*x + 12*(x^2) + 48*y - 36*x*y + 27*(y^2)));
	return z;
}

int main() {
	Variable x = random_variable({2,3}, "x");
	std::cout << _dot_var(x);
	std::cout << _dot_var(x, true);
	x.show();

	Variable x0 = Variable({1});
	Variable x1 = Variable({1});
	Variable y = x0 + x1;
	std::string txt = _dot_func(y.get_creator().get());
	std::cout << txt;

	Variable x2 = Variable({1}, "x");
	Variable y2 = Variable({1}, "y");
	Variable z2 = goldstein(x2, y2);
	z2.backward();
	z2.set_name("z");
	plot_dot_graph(z2, false, "goldstein");

}
