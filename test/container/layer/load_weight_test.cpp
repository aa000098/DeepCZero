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

	// 파일 전체 읽기
	file.seekg(0, std::ios::end);
	size_t filesize = file.tellg();
	file.seekg(0, std::ios::beg);

	std::vector<uint8_t> buffer(filesize);
	file.read(reinterpret_cast<char*>(buffer.data()), filesize);

	// 예시: 첫 16바이트 출력
	for (int i = 0; i < 16; ++i) {
		std::printf("%02X ", buffer[i]);
	}
	std::puts("");



}
