#include "deepczero.hpp"

#include <memory>
#include <vector>

int main() {
	Variable x({1});
	x.set_name("x");
	Variable y = sin(x);
	y.set_name("y");
	y.backward(false, true);

	std::vector<Tensor<>> logs;
	logs.push_back(y.data());

	Variable gx;
	for (int i = 0; i < 3; i++) {
		logs.push_back(x.grad().data());

		gx = x.grad();
		gx.set_name("gx");
		x.cleargrad();
		gx.backward(false, true);
		x.show();
	}

	gx = x.grad();
	gx.set_name("gx");

	plot_dot_graph(gx, false, "sin_higher_order_diff");
	trace_variable_refs(gx);
	gx.clear_graph();


	// sin graph visualization should use matplotlib
}
