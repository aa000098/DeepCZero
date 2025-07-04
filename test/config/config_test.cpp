#include "deepczero.hpp"

#include <iostream>

int main() {
	{
		dcz::no_grad();

		Variable x({2, 3, 4});
		Variable y = square(x);
		y.backward();
		std::cout << "Config::enable_backprop = false" << std::endl;
		y.show();
		x.show();
	}

	Variable x2 = Variable({100, 101, 102});
	Variable y2 = square(x2);
	std::cout << "Config::enable_backprop = true" << std::endl;
	y2.backward();
	y2.show();
	x2.show();

}
