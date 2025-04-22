/*#include "container/variable.hpp"
#include "container/tensor/tensorND.hpp"
#include "ops/ops.hpp"

#include <iostream>

using namespace tensor;

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
	
	TensorND t({2,2}, {3, 4, 5, 6});
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

	TensorND t2({2, 3, 4}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24});
	Variable c(t2);
	std::cout << "Variable c({3, 4}, {1,2,3,4,5,6,7,8,9,10,11,12})" << std::endl;
	std::cout << "c.get_shape() : ";
	for (size_t i=0; i< c.ndim(); i++) 
		std::cout << c.shape()[i] << " "; 
	std::cout << std::endl;
	std::cout << "c.size() : " << c.size() << std::endl;
	std::cout << "c.ndim() : " << c.ndim() << std:: endl;
	std::cout << "c.show() : \n" << std::endl;
	c.show();


}
*/
