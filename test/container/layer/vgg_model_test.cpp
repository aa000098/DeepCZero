#include "deepczero.hpp"

void test_vgg_forward_backward() {
	Tensor<> x_data({1, 3, 64, 64}, 1);
	Variable x(x_data);
	VGG16 model(true);	
	Variable y = model.forward({x});
	y.show();
}
                
int main() {
	test_vgg_forward_backward();
}
