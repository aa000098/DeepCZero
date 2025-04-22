#include "container/tensor/tensor1D.hpp"
// #include "ops/ops.hpp"

#include <iostream>

using namespace tensor;

int main() {
	Tensor1D<float> a({3, 4, 5});

	std::cout << "Tensor1D a({3, 4, 5})" << std::endl;
	std::cout << "a.shape() : ";
	for (size_t i=0; i< a.ndim(); i++) 
		std::cout << a.shape()[i] << " "; 
	std::cout << std::endl;
	std::cout << "a.size() : " << a.size() << std::endl;
	std::cout << "a.ndim() : " << a.ndim() << std:: endl;
//	std::cout << "a[1] : " << a[1] << std::endl;

//	std::cout << "a.show() : \n" << std::endl;
//	a.show();
}	
