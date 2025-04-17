#include "container/variable.hpp"
#include "ops/ops.hpp"
#include "config/config.hpp"

#include <iostream>

int main() {
	{
		dcz::UsingConfig no_grad(false);

		Variable x({2, 3, 4});
		Variable y = square(x);
		y.backward();
		std::cout << "Config::enable_backprop = false" << std::endl;
		y.show();
		x.show();
	}

	Variable x = Variable({100, 101, 102});
	Variable y = square(x);
	std::cout << "Config::enable_backprop = true" << std::endl;
	y.backward();
	y.show();
	x.show();

}
