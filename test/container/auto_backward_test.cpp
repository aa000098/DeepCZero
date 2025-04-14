#include "ops/ops.hpp"
#include "container/variable.hpp"

#include <iostream>

int main() {
	Variable x({0.5});

	Variable y = add(square(exp(square(x))), x);
	
	y.backward();

	std::cout << "[add(square(exp(square(0.5))), 0.5)]: ";
	y.show();
	std::cout << "[x auto backward]: ";
	x.show();

	Variable a({2});
	Variable b({3});
	Variable z = add(square(a), square(b));
	z.backward();
	std::cout << "[add(square(a), square(b))]: ";
	z.show();
	std::cout << "[a auto backward]: ";
	a.show();
	std::cout << "[b auto backward]: ";
	b.show();
}
