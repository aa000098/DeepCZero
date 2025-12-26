#include "deepczero.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <cstdlib>      // std::getenv
#include <filesystem>   // C++17
#include <stdexcept>

static std::filesystem::path get_home_dir() {
    if (const char* home = std::getenv("HOME")) {
        return std::filesystem::path(home);
    }
    // HOME이 비어있는 특이 케이스 대비(대부분 Linux에선 HOME 있음)
    throw std::runtime_error("HOME environment variable is not set");
}

void load_onnx_model_test() {
    const std::filesystem::path model_path = get_home_dir()/"Desktop"/"project"/"DeepCZero"/"models"/"yolov8n_fixed.onnx";  
    std::ifstream file(model_path.string(), std::ios::binary);

    if (!file) {
        std::cerr << "Failed to open file: " << model_path.string() << std::endl;
        return ;
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
}

int main() {
/*    try {
        load_onnx_model_test();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    */
    return 0;
}
