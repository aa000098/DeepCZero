#include "deepczero.hpp"

#include <fstream>
#include <vector>
#include <iostream>


int main() {

	std::string filename = "/home/user/project/DeepCZero/models/yolov8n.pt";
	std::ifstream file(filename, std::ios::binary);
	if (!file) {
		std::cerr << "File cannot opend: " << filename << std::endl;
		return 0;
	}


	std::string txt;
	while(getline(file, txt)) {
		std::cout << txt << std::endl;
	}

}
