#include <deepczero.hpp>

#include "cnpy.h"
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

void test_load_weights_npz() {
	VGG16 model(true);

	const std::filesystem::path weights_path = get_home_dir() / ".deepczero" / "weights" / "vgg16.npz";

    if (!std::filesystem::exists(weights_path)) {
        throw std::runtime_error("Weight file not found: " + weights_path.string());
    }


    // 1) npz 파일 로드
    cnpy::npz_t npz = cnpy::npz_load(weights_path.string());

    std::cerr << "[VGG16] npz contains " << npz.size() << " arrays:\n";

    // 2) 키 / shape / 워드 사이즈 출력
    for (const auto& kv : npz) {
        const std::string& key = kv.first;
        const cnpy::NpyArray& arr = kv.second;

        std::cerr << "  key: " << key << " | shape = (";

        for (size_t i = 0; i < arr.shape.size(); ++i) {
            std::cerr << arr.shape[i];
            if (i + 1 < arr.shape.size()) std::cerr << ", ";
        }
        std::cerr << ")";

        std::cerr << " | word_size = " << arr.word_size << " bytes";

        std::cerr << "\n";
    }

    std::cerr << "[VGG16] weight file loaded (no assignment yet)\n";
}

int main() {
	test_load_weights_npz();	
	return 0;
}
