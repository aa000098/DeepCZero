#include "deepczero.hpp"

#include <cmath>
#include <iostream>

Variable my_sin(Variable x, float threshold=1e-45) {
	Variable y({0});
	
	for (size_t i = 0; i < 100000; ++i) {
		float maclaurin = static_cast<float>(std::pow(-1, i) / std::tgamma(2*i+2));
		Variable c({maclaurin});
		Variable t = c * (x^(2*i+1));		
		y.set_name("y" + std::to_string(i));
		y = y + t;
	/*	
		std::cout << "[" << i << "] : ";
		std::cout << std::abs(t.data().raw_data()[0]) << "*" << maclaurin << std::endl;
		y.show();
	*/
		if (std::abs(t.data().raw_data()[0]) < threshold)
			break;
	}
	return y;
}

int main() {
	float pi = std::acos(-1.0);

	Variable x({pi/4});
	Variable y = my_sin(x);
	y.backward();
	y.set_name("y");
	x.set_name("x");
	y.show();
	x.show();
	plot_dot_graph(y, false, "my_sin");
}
