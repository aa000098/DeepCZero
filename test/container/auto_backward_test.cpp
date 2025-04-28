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

	Variable x6 = Variable({3,4});
	Variable x7 = Variable({2,3});
	Variable x8 = square(mul(square(x6), square(x7)));
	std::cout << "[Square(Mul(Square({3,4}), Square({2,3})))] ";
	x8.backward();
	x8.show();
	x6.show();
	std::cout << "[a auto backward]: ({1728, 20736})";
	x7.show();
	std::cout << "[b auto backward]: ({2592, 27648})";

}
