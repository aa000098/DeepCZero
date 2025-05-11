#include "container/variable_all.hpp"
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
	std::cout << "[a({2}) + 2]: " << std::endl;
	Variable test_2 = test_1 + 2;
	test_2.show();
	std::cout << "[3 + a({4})]: " << std::endl;
	Variable test_3 = 3 + test_2;
	test_3.show();
	std::cout << "[a({2}) * 2]: " << std::endl;
	Variable test_4 = test_1 * 2;
	test_4.show();
	std::cout << "[3 * a({2})]: " << std::endl;
	Variable test_5 = 3 * test_1;
	test_5.show();

	Variable test_6({3});
	std::cout << "[-a({3})]: " << std::endl;
	Variable test_7 = -test_6;
	test_7.show();
	std::cout << "[a({3}) - b({-3})]: " << std::endl;
	Variable test_8 = test_6 - test_7;
	test_8.show();
	std::cout << "[-3 - c({6} - 3)]: " << std::endl;
	Variable test_9 = -3 - test_8 - 3;
	test_9.show();
	std::cout << "[c({-12}) / a({3})]: " << std::endl;
	Variable test_10 = test_9 / test_6;
	test_10.show();
	std::cout << "[10 / (d({-4} / 2))]: " << std::endl;
	Variable test_11 = 10 / (test_10 / 2);
	test_11.show();
	std::cout << "[e({-5})^2]: " << std::endl;
	Variable test_12 = test_11^2;
	test_12.show();
	
	std::cout << "[x.reshape({3,2})]: " << std::endl;
	Tensor t1({6}, 1.0f);
	Variable test13(t1);
	Variable test14 = test13.reshape({3,2});
	test13.show();
	test14.show();

	std::cout << "[x.transpose({2,3,4,5})]: " << std::endl;
	Tensor t2({2,3,4,5}, 1.0f);
	Variable test15(t2);
	Variable test16 = test15.trans({1,0,3,2});
	std::cout << "original shape: ";
	for (size_t i = 0; i < test15.ndim(); i++) 
		std::cout << test15.shape()[i] << " ";
	std::cout << std::endl;
	std::cout << "transposed shape: ";
	for (size_t i = 0; i < test16.ndim(); i++) 
		std::cout << test16.shape()[i] << " ";
	std::cout << std::endl;
}
