#include "container/tensor/tensorND.hpp"

#include <iostream>

using namespace tensor;

int main() {

	TensorND<float> a({3,2}, {3, 4, 5, 6, 7, 8});

	std::cout << "TensorND a({3, 2}, {3, 4, 5, 6, 7, 8})" << std::endl;
	a.show();

	std::cout << "a.get_shape() : "; 
	for (auto shape : a.get_shape())
		std::cout <<  shape << " ";
	std::cout << std::endl; 
	
	std::cout << "a.size() : " << a.size() << std::endl;
	std::cout << "a.ndim() : " << a.ndim() << std:: endl;
	std::cout << "a.get_strides() : ";
	for (auto s : a.get_strides())
		std::cout <<  s << " ";
	std::cout << std::endl; 
	std::cout << "a({0, 1}) : " << a(std::vector<size_t>{0, 1}) << std::endl;
	std::cout << "a({2, 0}) : " << a(std::vector<size_t>{2, 0}) << std::endl;
//	std::cout << "a[1] : " << a[1] << std::endl;

	std::cout << std::endl;

	
	TensorND<float> b({4,3,2}, 1.0f);
	std::cout << "TensorND b({4, 3, 2}, 1.0f)" << std::endl;
	b.show();
	
	b({2, 1, 1}) = 3;
	std::cout << "b({2, 1, 1}) = 3" << std::endl;
	b.show();



/*	Variable b(t);

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
*/
}
