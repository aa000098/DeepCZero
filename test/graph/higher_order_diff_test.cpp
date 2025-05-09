#include "deepczero.hpp"

#include <string>

int main() {
	Variable x({1});
	Variable y = tanh(x);
	x.set_name("x");
	y.set_name("y");

	y.backward(false,true);

	int iters = 2;
	for (int i = 0; i < iters; i++) {
		std::string name = std::string("gx") + std::to_string(iters+1);
		Variable gx = x.grad();
		gx.set_name(name);
		x.cleargrad();
		gx.backward(false, true);
	}

	std::string name = std::string("gx") + std::to_string(iters+1);
	Variable gx = x.grad();
	gx.set_name(name);
	plot_dot_graph(gx, false, "tanh");
}
