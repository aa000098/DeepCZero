#include "container/tensor/tensorview.hpp"
#include "container/tensor/tensorND.hpp"

using namespace tensor;

int main() {
	TensorND<float> tensor({4, 3, 2});


	TensorView<float> view({2}, tensor.shared_data(), {1});

	std::cout << "TensorView view({2}, tensor.shared_data(), {1})" << std::endl;
	view.show();
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

	view({1}) = 2;
	std::cout << "view({1}) = 2" << std::endl;
	std::cout << "view.show()" <<std::endl;
	view.show();
	std::cout << "tensor.show()" <<std::endl;
	tensor.show();
	
	tensor({0,0,0}) = 4;
	std::cout << "tensor({0,0,0}) = 4" << std::endl;
	std::cout << "view.show()" <<std::endl;
	view.show();
	std::cout << "tensor.show()" <<std::endl;
	tensor.show();
}
