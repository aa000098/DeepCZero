
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

	// Test zeros_like, ones_like, full_like
	std::cout << "\n=== zeros_like / ones_like / full_like ===" << std::endl;

	Tensor src({2, 3}, 5.0f);
	std::cout << "src({2,3}, 5.0):" << std::endl;
	src.show();

	auto z = Tensor<>::zeros_like(src);
	std::cout << "zeros_like(src):" << std::endl;
	z.show();
	std::cout << "shape match: " << (z.get_shape() == src.get_shape() ? "OK" : "FAIL") << std::endl;
	std::cout << "value check (expect 0): " << z({0, 0}) << std::endl;

	auto o = Tensor<>::ones_like(src);
	std::cout << "ones_like(src):" << std::endl;
	o.show();
	std::cout << "shape match: " << (o.get_shape() == src.get_shape() ? "OK" : "FAIL") << std::endl;
	std::cout << "value check (expect 1): " << o({0, 0}) << std::endl;

	auto f = Tensor<>::full_like(src, 3.14f);
	std::cout << "full_like(src, 3.14):" << std::endl;
	f.show();
	std::cout << "shape match: " << (f.get_shape() == src.get_shape() ? "OK" : "FAIL") << std::endl;
	std::cout << "value check (expect 3.14): " << f({0, 0}) << std::endl;

	// Test with 1D tensor
	Tensor src1d(5, 9.0f);
	auto z1d = Tensor<>::zeros_like(src1d);
	std::cout << "\nzeros_like(1D, size=5):" << std::endl;
	z1d.show();
	std::cout << "shape match: " << (z1d.get_shape() == src1d.get_shape() ? "OK" : "FAIL") << std::endl;

}
