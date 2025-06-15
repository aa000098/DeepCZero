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

void test_spiral_to_csv_and_load() {
    // SpiralDataset 초기화
    size_t num_data = 100;
    size_t num_class = 3;
    bool train = true;
    SpiralDataset dataset(num_data, num_class, train);

    Tensor data = dataset.get_data();    // shape: [300, 2]
    Tensor label = dataset.get_label();  // shape: [300]

    // 저장 경로
    const char* home = std::getenv("HOME");
    if (!home) throw std::runtime_error("HOME environment variable is not set.");
    std::string dir_path = std::string(home) + "/.deepczero/datasets/";

    std::string data_filename = "spiral_data_with_header_and_index.csv";
    std::string label_filename = "spiral_label_with_header_and_index.csv";

    // CSV로 저장
    data.to_csv(data_filename, true, true);
    label.to_csv(label_filename, true, true);

    std::cout << "Spiral dataset saved to:\n"
              << "  " << dir_path + data_filename << "\n"
              << "  " << dir_path + label_filename << "\n";

    // 다시 로드해서 확인
    Tensor loaded_data = Tensor<float>::from_csv(data_filename, true, true);
    Tensor loaded_label = Tensor<float>::from_csv(label_filename, true, true);

    // Shape 출력
    const auto& data_shape = loaded_data.get_shape();
    const auto& label_shape = loaded_label.get_shape();

    std::cout << "Loaded data shape: ";
    for (auto d : data_shape) std::cout << d << " ";
    std::cout << "\n";

    std::cout << "Loaded label shape: ";
    for (auto d : label_shape) std::cout << d << " ";
    std::cout << "\n";

    // 간단한 값 확인 (optional)
    std::cout << "First row of data: ";
    for (size_t i = 0; i < data_shape[1]; ++i)
        std::cout << loaded_data({0, i}) << " ";
    std::cout << "\n";

    std::cout << "First label: " << loaded_label({0}) << "\n";
}


int main() {
	test_spiral_to_csv();
    test_spiral_to_csv_and_load();
}

