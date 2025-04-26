#include "container/variable.hpp"
#include "container/tensor/tensor.hpp"

#include <iostream>
#include <memory>

int main() {
	Variable my_var =	Variable({8}, "test", true);
	std::cout << "[Variable({8}, \"test\", true)]: " << std::endl;
	my_var.show();
	
	Variable my_var2 =	Variable({5,6,7}, "test", true);
	std::cout << "[Variable({5,6,7}, \"test\", true)]: " << std::endl;
	my_var2.show();


	Tensor ys = Tensor({2,3,4});
	auto out = std::make_shared<VariableImpl<>>(ys);
}
