#include "deepczero.hpp"

#include <iostream>

int main() {
	Variable x({0.5});

	Variable y1 = square(x);
	Variable y2 = exp(y1);
	Variable y3 = square(y2);
	y3.show();

	Tensor<float> x2({1}, 1);
	std::shared_ptr<Function> C = y3.get_creator();
	std::vector<Variable> grad_C = C->backward(x2);
	std::cout << "[grad_C]: "; 
	grad_C[0].show();
	std::cout << std::endl;

	std::shared_ptr<Function> B = y2.get_creator();
	std::vector<Variable> grad_B = B->backward(grad_C[0]);
	std::cout << "[grad_B]: ";
	grad_B[0].show();
	std::cout << std::endl;

	std::shared_ptr<Function> A = y1.get_creator();
	std::vector<Variable> grad_A = A->backward(grad_B[0]);
	std::cout << "[grad_A]: ";
	grad_A[0].show();
	std::cout << std::endl;

	Tensor<float> x3({3,2}, 0.5);
	Variable y4 = square(x3);
	Variable y5 = exp(y4);
	Variable y6 = square(y5);
	y6.show();

	Tensor<float> x4({3,2}, 1);
	std::shared_ptr<Function> C2 = y6.get_creator();
	std::vector<Variable> grad_C2 = C2->backward(x4);
	std::cout << "[grad_C2]: "; 
	grad_C2[0].show();

	std::shared_ptr<Function> B2 = y5.get_creator();
	std::vector<Variable> grad_B2 = B2->backward(grad_C2[0]);
	std::cout << "[grad_B2]: "; 
	grad_B2[0].show();

	std::shared_ptr<Function> A2 = y4.get_creator();
	std::vector<Variable> grad_A2 = A2->backward(grad_B2[0]);
	std::cout << "[grad_A2]: "; 
	grad_A2[0].show();
}
