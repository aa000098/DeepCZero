#include "container/variable.hpp"

#include <iostream>

int main() {
	Variable my_var =	Variable({8}, true);
	std::cout << "[Variable({8}, true)]: ";
	my_var.show();
}
