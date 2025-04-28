#include "ops/ops.hpp"
#include "container/variable.hpp"

#include <iostream>

int main() {
	Variable x = Variable({8}); 
	Variable y1 = square(x);
	std::cout << "[Square(8)]: ";
	y1.show();


	Variable y2 = exp(x);
	std::cout << "[Exp(8)]: ";
	y2.show();

	Variable x2 = Variable({0.5});
	Variable y3 = square(x2);
	Variable y4 = exp(y3);
	Variable y5 = square(y4);
	std::cout << "[Square(Exp(Square(0.5)))]: ";
	y3.show();
	y4.show();
	y5.show();

	Variable x6 = Variable({3,4});
	Variable x7 = Variable({2,3});
	Variable x8 = square(mul(square(x6), square(x7)));
	std::cout << "[Square(Mul(Square({3,4}), Square({2,3})))] ";
	x8.show();

	std::cout << std::endl;
}
