#include "container/variable.hpp"

#include <iostream>

int main() {
	Variable my_var =	Variable({8}, "test", true);
	std::cout << "[Variable({8}, \"test\", true)]: " << std::endl;
	my_var.show();
}
