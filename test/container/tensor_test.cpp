#include "container/variable.hpp"
#include "container/tensor.hpp"
#include "ops/ops.hpp"

#include <iostream>

int main() {
	Variable a({3, 4, 5});

	std::cout << "Variable a({3, 4, 5})" << std::endl;
	std::cout << "a.get_shape() : ";
	for (size_t i=0; i< a.ndim(); i++) 
		std::cout << a.shape()[i] << " "; 
	std::cout << std::endl;
	std::cout << "a.size() : " << a.size() << std::endl;
	std::cout << "a.ndim() : " << a.ndim() << std:: endl;
//	std::cout << "a[1] : " << a[1] << std::endl;
	std::cout << "a.show() : \n" << std::endl;
	a.show();
	
	Tensor t({2,2}, {3, 4, 5, 6});
	Variable b(t);

	std::cout << "Variable b({2, 2}, {3, 4, 5, 6})" << std::endl;
	std::cout << "b.get_shape() : ";
	for (size_t i=0; i< b.ndim(); i++) 
		std::cout << b.shape()[i] << " "; 
	std::cout << std::endl;
	std::cout << "b.size() : " << b.size() << std::endl;
	std::cout << "b.ndim() : " << b.ndim() << std:: endl;
//	std::cout << "b[1] : " << b[1] << std::endl;
	std::cout << "b.show() : \n" << std::endl;
	b.show();
}
