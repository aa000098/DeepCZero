
#include "container/tensor/tensor.hpp"

#include <iostream>

using namespace tensor;

int main() {
	Tensor a({3, 4, 5}, 1);

	std::cout << "Tensor a({3, 4, 5}, 1)" << std::endl;
	a.show();
	std::cout << "a.get_shape() : ";
	for (size_t i=0; i< a.ndim(); i++) 
		std::cout << a.get_shape()[i] << " "; 
	std::cout << std::endl;
	std::cout << "a.size() : " << a.size() << std::endl;
	std::cout << "a.ndim() : " << a.ndim() << std:: endl;
	std::cout << "a[1] " << std::endl;
	a[1].show();
	std::cout << "a({0, 1, 2}) = 2" << std::endl;
	a({0, 1, 2}) = 2;
	std::cout << "a[0][1].show()" << std::endl;
	a[0][1].show();
	std::cout << "a.show()" << std::endl;
	a.show();

	
	Tensor b({2,3,4});
	std::cout << "Tensor b({2,3,4})" << std::endl;
	b.show();

	Tensor c(b);
	std::cout << "Tensor c(b)" << std::endl;
	c.show();

	Tensor d(4);
	std::cout << "Tensor d(4)" << std::endl;
	d.show();
	
	

}
