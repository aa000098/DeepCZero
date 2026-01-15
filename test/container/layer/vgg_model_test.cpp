#include "deepczero.hpp"

void test_vgg_forward_backward() {
	dcz::UsingConfig test_mode("train", false);  // dropout이 그래프 끊지 않도록

	// pretrained=true 사용 시 224x224 입력 필수
	// fc6 weight shape: (25088, 4096) = (512*7*7, 4096)
	// 64x64 입력 -> pooling 5회 -> 2x2 -> flatten = 2048 (mismatch)
	// 224x224 입력 -> pooling 5회 -> 7x7 -> flatten = 25088 (OK)
	Tensor<> x_data({1, 3, 224, 224}, 1);
	Variable x(x_data);
	VGG16 model(true);
	// Variable y = model.forward({x});
	// y.show();
	model.plot({x}, "vgg16");
}

int main() {
	test_vgg_forward_backward();
}
