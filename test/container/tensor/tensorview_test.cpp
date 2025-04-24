#include "container/tensor/tensorview.hpp"
#include "container/tensor/tensorND.hpp"

using namespace tensor;

int main() {
	TensorND<float> tensor({4, 3});


	TensorView<float> view({2, 6}, tensor.raw_data(), {5, 1});

	std::cout << "TensorView view({2, 6})" << std::endl;
	std::cout << "view.get_shape() : "; 
	for (auto shape : view.get_shape())
		std::cout <<  shape << " ";
	std::cout << std::endl; 
//	std::cout << "view[1] : " << view[1] << std::endl;
	std::cout << "view.size() : " << view.size() << std::endl;
	std::cout << "view.ndim() : " << view.ndim() << std:: endl;
	std::cout << "view.get_strides() : ";
	for (auto s : view.get_strides())
		std::cout <<  s << " ";
	std::cout << std::endl; 
	
	view.show();

}
