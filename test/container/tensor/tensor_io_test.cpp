#include "deepczero.hpp"

#include <cassert>
#include <iostream>

void test_tensor_to_csv() {
    using namespace tensor;

    // 1. 테스트용 텐서 생성
    Tensor<float> t({3, 2}, {1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f});

    // 2. CSV로 저장
    std::string filename = "test_tensor.csv";
    t.to_csv(filename, true, true);  // 인덱스, 헤더 포함

    // 3. 다시 로드
    Tensor<float> loaded = Tensor<float>::from_csv(filename, true, true);

    // 4. shape 동일한지 확인
    assert(loaded.get_shape() == t.get_shape());

    // 5. 데이터 값 동일한지 확인
    const auto& original_data = t.raw_data();
    const auto& loaded_data = loaded.raw_data();
    for (size_t i = 0; i < t.size(); ++i) {
        assert(std::abs(original_data[i] - loaded_data[i]) < 1e-6f);
    }

    std::cout << "[✅ PASSED] test_tensor_to_csv\n";
}

void test_tensor_from_csv() {
    using namespace tensor;

    std::string csv_file = "test_tensor.csv";

    // 1. Load tensor from CSV
    Tensor<> loaded = Tensor<>::from_csv(csv_file, 
								/*index=*/true, 
								/*header=*/true, 
								/*delimiter=*/',');

    // 2. 기본 shape 확인 (예: spiral_data.csv 는 2차원 텐서여야 함)
    const std::vector<size_t>& shape = loaded.get_shape();
    assert(shape.size() == 2);          // 2D tensor
    assert(shape[1] == 2);              // 2 features per sample

    // 3. 값 일부 확인 (예: 첫 값들)
    const std::vector<float>& data = loaded.raw_data();

    // spiral_data.csv가 다음 값으로 시작한다고 가정:
    // 0,0,-0.305973
    // 0,1,0.1896854
    // → 첫 벡터는 (-0.305973, 0.1896854)
    assert(std::abs(data[0] - (1.1f)) < 1e-5f);
    assert(std::abs(data[1] - 2.2f) < 1e-5f);

    std::cout << "[✅ PASSED] test_tensor_from_csv: CSV loaded and values verified." << std::endl;
}

int main() {
	test_tensor_to_csv();
	test_tensor_from_csv();
}
