#include "ops/ops.hpp"
#include "container/variable_all.hpp"

#include <iostream>

int main() {
	Variable x({0.5});
	Variable y = square(exp(square(x)));
	
	y.backward();

	std::cout << "[add(square(exp(square(0.5))), 0.5)]: "<< std::endl;
	y.show();
	std::cout << "[x auto backward]: (3.29744)" << std::endl;
	x.show();

	Variable a({2});
	Variable b({3});
	Variable z = square(add(square(a), square(b)));
	z.backward();
	std::cout << "[square(add(square(2), square(3)))]: " << std::endl;
	z.show();
	std::cout << "[a auto backward]: (104)" << std::endl;
	a.show();
	std::cout << "[b auto backward]: (156)" << std::endl;
	b.show();

	Variable x6 = Variable({3,4});
	Variable x7 = Variable({2,3});
	Variable x8 = square(mul(square(x6), square(x7)));
	std::cout << "[Square(Mul(Square({3,4}), Square({2,3})))]" << std::endl;
	x8.backward();
	x8.show();
	std::cout << "[a auto backward]: ({1728, 20736})" << std::endl;
	x6.show();
	std::cout << "[b auto backward]: ({2592, 27648})" << std::endl;
	x7.show();

	Variable x9 = Variable({2});
	Variable x10 = Variable({4});
	Variable x11 = -((x9^2.0f) - (x10^2.0f)) / 2;
	std::cout << "-(2^2 - 4^2) / 2" << std::endl;

	x11.backward();
	x11.show();
	std::cout << "[b auto backward]: ({, })" << std::endl;
	x10.show();
	std::cout << "[a auto backward]: ({, })" << std::endl;
	x9.show();

}
