#include "deepczero.hpp"

void test_vgg_forward_backward() {
	Tensor<> x_data({1, 3, 244, 244}, 1);
	Variable x(x_data);
	VGG16 model;	
	Variable y = model.forward({x});
}
               
int main() {
	test_vgg_forward_backward();
}
