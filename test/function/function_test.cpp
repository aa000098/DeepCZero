#include "function/function.hpp"
#include "container/variable.hpp"

#include <iostream>

int main() {
	Variable x = Variable(8); 
	Square f1;
	Variable y1 = f1(x);
	std::cout << "[Square(8)]: ";
	y1.show();


	Exp f2;
	Variable y2 = f2(x);
	std::cout << "[Exp(8)]: ";
	y2.show();

	Variable x2 = Variable(0.5);
	Square f3;
	Exp f4;
	Square f5;
	Variable y3 = f3(x2);
	Variable y4 = f4(y3);
	Variable y5 = f5(y4);
	std::cout << "[Square(Exp(Square(0.5)))]: ";
	y3.show();
	y4.show();
	y5.show();


	std::cout << std::endl;
}
