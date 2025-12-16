#include "deepczero.hpp"

#include <iostream>


void test_load_weights_vgg16() {
	Tensor<> x_data({1, 3, 64, 64}, 1);
	Variable x(x_data);
	VGG16 model(true);	


//	Variable y = model.forward({x});

}

int main() {
	test_load_weights_vgg16();
	return 0;
}
