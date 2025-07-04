#include "deepczero.hpp"

#include <fstream>
#include <iostream>

int main() {
    std::string filename = "/home/user/project/DeepCZero/models/yolov8n_fixed.onnx";  // ← 여기에 ONNX 파일 경로 넣으세요
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return 1;
    }

    const int bytes_per_line = 16;
    unsigned char buffer[bytes_per_line];
    int offset = 0;

    while (file.read(reinterpret_cast<char*>(buffer), bytes_per_line) || file.gcount() > 0) {
        std::cout << std::hex << std::setw(8) << std::setfill('0') << offset << ": ";

        for (int i = 0; i < file.gcount(); ++i)
            std::cout << std::setw(2) << static_cast<int>(buffer[i]) << " ";

        std::cout << std::endl;
        offset += file.gcount();
    }

    file.close();
    return 0;
}
