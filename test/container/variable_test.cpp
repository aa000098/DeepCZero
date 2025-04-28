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
	std::cout << "[VariableImpl(Tensor({2,3,4})), \"test\", true)]: " << std::endl;
	auto out = std::make_shared<VariableImpl<>>(ys);
	Variable my_var3 = Variable(out);
	my_var3.show();

	Variable test_a({2,3});
	Variable test_b({2,3});
	std::cout << "[a({2,3}) + b({2,3})]: " << std::endl;
	Variable test_c = test_a + test_b;
	test_c.show();
	std::cout << "[a({2,3}) * c({4,6})]: " << std::endl;
	Variable test_d = test_a * test_c;
	test_d.show();
	std::cout << "[{a({2,3}) + b({2,3})} * d({8,18})]: " << std::endl;
	Variable test_e = (test_a + test_b) * test_d;
	test_e.show();

	Variable test_1({2});
	Variable test_2 = test_1 + 2;
	Variable test_3 = 3 + test_2;
	Variable test_4 = test_1 * 2;
	Variable test_5 = 3 * test_1;
	test_2.show();
	test_3.show();
	test_4.show();
	test_5.show();
}
