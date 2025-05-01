#include "function/function.hpp"
#include "container/variable.hpp"
#include "container/tensor/tensor.hpp"

#include <iostream>

int main() {
	Variable x = Variable({8}); 
	std::shared_ptr<Function> f1 = std::make_shared<Square>();
	Variable y1 = (*f1)({x});
	std::cout << "[Square(8)]: ";
	y1.show();
	std::cout << std::endl;


	std::shared_ptr<Function> f2 = std::make_shared<Exp>();
	Variable y2 = (*f2)({x});
	std::cout << "[Exp(8)]: ";
	y2.show();
	std::cout << std::endl;

	Variable x2 = Variable({0.5});
	std::shared_ptr<Function> f3 = std::make_shared<Square>();
	std::shared_ptr<Function> f4 = std::make_shared<Exp>();
	std::shared_ptr<Function> f5 = std::make_shared<Square>();
	Variable y3 = (*f3)({x2});
	Variable y4 = (*f4)({y3});
	Variable y5 = (*f5)({y4});
	std::cout << "[Square(0.5)]: ";
	y3.show();
	std::cout << "[Exp(Square(0.5))]: ";
	y4.show();
	std::cout << "[Square(Exp(Square(0.5)))]: ";
	y5.show();

	f1.reset();
	f2.reset();
	f3.reset();
	f4.reset();
	f5.reset();


	std::cout << std::endl;

	Tensor<float> t({3,2}, 2); 
	Variable x3 = Variable(t);
	std::shared_ptr<Function> f6 = std::make_shared<Square>();
	Variable y6 = (*f6)({x3});
	std::cout << "[Square(Tensor({3,2}))]: ";
	y6.show();
	std::cout << std::endl;
	f6.reset();

	std::shared_ptr<Function> mul_f = std::make_shared<Mul>();
	Tensor<float> t2({2,3}, 3);
	Tensor<float> t3({2,3}, 2);
	Variable y7 = (*mul_f)({t2,t3});
	std::cout << "[Mul(Tensor({2,3},3), Tensor({2,3},2))]: ";
	y7.show();
	std::cout << std::endl;
	mul_f.reset();

	std::shared_ptr<Function> neg_f = std::make_shared<Neg>();
	Variable y8 = (*neg_f)({t2});
	std::cout << "[Neg(Tensor({2,3},3))]: ";
	y8.show();
	std::cout << std::endl;
	neg_f.reset();
	
	std::shared_ptr<Function> sub_f = std::make_shared<Sub>();
	Variable y9 = (*sub_f)({t2, t3});
	std::cout << "[Sub(Tensor({2,3},3), Tensor({2,3},2))]: ";
	y9.show();
	std::cout << std::endl;
	sub_f.reset();	

	std::shared_ptr<Function> div_f = std::make_shared<Div>();
	Variable y10 = (*div_f)({t2, t3});
	std::cout << "[Div(Tensor({2,3},3), Tensor({2,3},2))]: ";
	y10.show();
	std::cout << std::endl;
	div_f.reset();	

	std::shared_ptr<Function> pow_f = std::make_shared<Pow>();
	Variable y11 = (*pow_f)({t2, t3});
	std::cout << "[Pow(Tensor({2,3},3), Tensor({2,3},2))]: ";
	y11.show();
	std::cout << std::endl;
	pow_f.reset();	

}
