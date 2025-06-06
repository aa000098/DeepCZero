#include "deepczero.hpp"

#include <cassert>
#include <iostream>


using namespace tensor;

// 테스트 함수
void test_get_item_forward_and_backward() {
	std::cout << "[Test] get_item + forward + backward\n";

	// 입력 Tensor x: 3x4, requires_grad=true
	Tensor<float> data({3, 4}, {
		1, 2, 3, 4,   // row 0
		5, 6, 7, 8,   // row 1 ← 우리가 가져올 부분
		9, 10, 11, 12 // row 2
	});
	Variable x(data);

	// slicing: x[1]
	std::vector<size_t> slices = {1};
	Variable y = get_item(x, slices);

	y.show();

	assert(y({0}) == 5.0f);
	assert(y({1}) == 6.0f);
	assert(y({2}) == 7.0f);
	assert(y({3}) == 8.0f);

	// 역전파
	y.backward();

	// 결과 확인
	const Tensor<float>& dx = x.grad().data();
	dx.show();

	// 기대 결과: x.grad()[1] == [1, 1, 1, 1]
	const auto& dx_data = dx.raw_data();
	assert(dx_data[4] == 1.0f);  // x[1][0]
	assert(dx_data[5] == 1.0f);  // x[1][1]
	assert(dx_data[6] == 1.0f);  // x[1][2]
	assert(dx_data[7] == 1.0f);  // x[1][3]

	// 나머지는 0
	assert(dx_data[0] == 0.0f);
	assert(dx_data[1] == 0.0f);
	assert(dx_data[2] == 0.0f);
	assert(dx_data[3] == 0.0f);
	assert(dx_data[8] == 0.0f);
	assert(dx_data[9] == 0.0f);
	assert(dx_data[10] == 0.0f);
	assert(dx_data[11] == 0.0f);

	std::cout << "✅ test_get_item_forward_and_backward passed.\n";
}

int main() {
	test_get_item_forward_and_backward();
	return 0;
}

