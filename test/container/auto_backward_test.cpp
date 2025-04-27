#include "ops/ops.hpp"
#include "container/variable.hpp"

#include <iostream>

int main() {
	Variable x({0.5});
	Variable y = square(exp(square(x)));
	
	y.backward();

	std::cout << "[add(square(exp(square(0.5))), 0.5)]: ";
	y.show();
	std::cout << "[x auto backward]: (3.29744)";
	x.show();

	Variable a({2});
	Variable b({3});
	Variable z = square(add(square(a), square(b)));
	z.backward();
	std::cout << "[square(add(square(2), square(3)))]: ";
	z.show();
	std::cout << "[a auto backward]: (104)";
	a.show();
	std::cout << "[b auto backward]: (156)";
	b.show();
}
