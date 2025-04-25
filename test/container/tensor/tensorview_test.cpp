#include "container/tensor/tensorview.hpp"
#include "container/tensor/tensorND.hpp"

using namespace tensor;

int main() {
	TensorND<float> tensor({4, 3, 2});

	std::cout << "tensor({4, 3, 2})" << std::endl;
	std::cout << "tensor.get_shape() : "; 
	for (auto shape : tensor.get_shape())
		std::cout <<  shape << " ";
	std::cout << std::endl;
	std::cout << "tensor.get_strides() : ";
	for (auto s : tensor.get_strides())
		std::cout <<  s << " ";
	std::cout << std::endl;

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
	tensor({1,0,1}) = 3;
	tensor({2,1,1}) = 5;
	std::cout << "tensor({0,0,0}) = 4" << std::endl;
	std::cout << "tensor({1,0,1}) = 3" << std::endl;
	std::cout << "tensor({2,1,2}) = 5" << std::endl;
	std::cout << "view.show()" <<std::endl;
	view.show();
	std::cout << "tensor.show()" <<std::endl;
	tensor.show();

	std::cout << "tensor[1]" << std::endl;
	tensor[1].show();
	std::cout << "tensor[2][1]" << std::endl;
	tensor[2][1].show();
}
