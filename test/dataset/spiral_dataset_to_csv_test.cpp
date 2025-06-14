#include "deepczero.hpp"

#include <cstdlib>
#include <string>

void test_spiral_to_csv() {
    // SpiralDataset 초기화
    size_t num_data = 100;
    size_t num_class = 3;
	bool train = true;
    SpiralDataset dataset(num_data, num_class, train);

    Tensor data = dataset.get_data();
    Tensor label = dataset.get_label();

    // 저장할 디렉토리 생성
    const char* home = std::getenv("HOME");
    if (!home) {
        throw std::runtime_error("HOME environment variable is not set.");
    }

    std::string dir_path = std::string(home) + "/.deepczero/datasets";

    // 파일 경로
	std::string data_filename = "spiral_data.csv";
	std::string label_filename = "spiral_label.csv";
    std::string data_file = dir_path + "/" + data_filename; 
    std::string label_file = dir_path + "/" + label_filename;;

    // CSV 파일로 저장
    data.to_csv(data_filename, false, false);
    label.to_csv(label_filename, false, false);

    std::cout << "Spiral dataset saved to:\n"
              << "  " << data_file << "\n"
              << "  " << label_file << "\n";
}

int main() {
	test_spiral_to_csv();
}

