#include "deepczero.hpp"

#include <cassert> 

void test_accuracy_all_correct() {
    // 예: 3개 샘플, 클래스 0~2 중 예측
	std::vector<float> vec(
        {0.9f, 0.05f, 0.05f,  // → class 0
        0.1f, 0.8f, 0.1f,    // → class 1
        0.0f, 0.3f, 0.7f});     // → class 2
    Tensor<float> y_data({3,3}, vec);

    Tensor<float> t_data({3}, {0, 1, 2});  // 정답 레이블

    Variable y_var(y_data);
    Variable t_var(t_data);

    function::Accuracy acc;
    Variable result = acc.forward(y_var, t_var);

    float acc_value = result.data()({0});
    std::cout << "Accuracy (all correct): " << acc_value << std::endl;
    assert(std::abs(acc_value - 1.0f) < 1e-6);
}

int main() {
	test_accuracy_all_correct();
}
