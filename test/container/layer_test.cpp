#include "deepczero.hpp"

#include <iostream>

using namespace layer;

void test_layer_register_and_get_param() {
    // 1. 파라미터 생성 및 값 할당
    Parameter param({1.0f, 2.0f, 3.0f});

    // 2. Layer 객체 생성 및 등록
	Linear layer;
    layer.register_params("weight", param);

    // 3. 등록된 파라미터 조회 및 출력
    try {
        Parameter retrieved = layer.get_param("weight");
        std::cout << "[TEST] Parameter 'weight' found:\n";
        retrieved.data().show();  // 내부 텐서 출력
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception occurred: " << e.what() << std::endl;
    }

    // 4. 존재하지 않는 파라미터 조회 시도
    try {
        layer.get_param("bias");  // 존재하지 않음
    } catch (const std::exception& e) {
        std::cerr << "[EXPECTED ERROR] " << e.what() << std::endl;
    }
}

int main() {
    test_layer_register_and_get_param();
    return 0;
}

