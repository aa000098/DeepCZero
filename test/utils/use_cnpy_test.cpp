#include "cnpy.h"
#include <iostream>
#include <deepczero.hpp>
#include <string>

void test_load_weights_npz() {
	VGG16 model(true);

	std::string weights_path = "/home/user/.deepczero/vgg16.npz";
    std::cerr << "[VGG16] loading weights from: " << weights_path << "\n";

    // 1) npz 파일 로드
    cnpy::npz_t npz = cnpy::npz_load(weights_path);

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
