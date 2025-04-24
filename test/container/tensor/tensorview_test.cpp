#include "container/tensor/tensorview.hpp"
#include "container/tensor/tensorND.hpp"

using namespace tensor;

int main() {
	TensorND<float> tensor({4, 3});
	std::shared_ptr<std::vector<float>> data = std::make_shared<std::vector<float>>(tensor.raw_data());
	
	TensorView<float> a(tensor.get_shape(), tensor.get_strides(), data);	

	TensorView<float> b(tensor.raw_data(), tensor.get_shape(), tensor.get_strides(), 0);



}
