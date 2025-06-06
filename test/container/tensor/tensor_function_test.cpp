#include "deepczero.hpp"

#include <cassert>
#include <iostream>

void test_tensor_sum() {
    std::cout << "[Test] Sum" << std::endl;
    Tensor<float> x({2, 2, 3}, {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10,11,12
    });

    // axis=0 → shape = [2, 3]
    Tensor<float> s0 = x.sum({0});
    std::vector<float> expected_s0 = {
        8, 10, 12,
        14, 16, 18
    };
    assert(s0.get_shape() == std::vector<size_t>({2, 3}));
    for (size_t i = 0; i < expected_s0.size(); ++i)
        assert(std::abs(s0.raw_data()[i] - expected_s0[i]) < 1e-5);

    // axis=1 → shape = [2, 3]
    Tensor<float> s1 = x.sum({1});
    std::vector<float> expected_s1 = {
        5, 7, 9,
        17, 19, 21
    };
    assert(s1.get_shape() == std::vector<size_t>({2, 3}));
    for (size_t i = 0; i < expected_s1.size(); ++i)
        assert(std::abs(s1.raw_data()[i] - expected_s1[i]) < 1e-5);

    // axis=2 → shape = [2, 2]
    Tensor<float> s2 = x.sum({2});
    std::vector<float> expected_s2 = {
        6, 15,
        24, 33
    };
    assert(s2.get_shape() == std::vector<size_t>({2, 2}));
    for (size_t i = 0; i < expected_s2.size(); ++i)
        assert(std::abs(s2.raw_data()[i] - expected_s2[i]) < 1e-5);

    // axis={0, 2} → shape = [2]
    Tensor<float> s02 = x.sum({0, 2});
    std::vector<float> expected_s02 = {
        30, 48
    };
    assert(s02.get_shape() == std::vector<size_t>({2}));
    for (size_t i = 0; i < expected_s02.size(); ++i)
        assert(std::abs(s02.raw_data()[i] - expected_s02[i]) < 1e-5);

    // sum over all elements
    Tensor<float> s_all = x.sum();  // axis omitted
    std::vector<float> expected_all = {78};
    assert(s_all.get_shape() == std::vector<size_t>{1});  // scalar
    assert(std::abs(s_all.raw_data()[0] - expected_all[0]) < 1e-5);

    std::cout << "✅ tensor sum tests passed.\n" << std::endl;
}

void test_broadcast_to() {
    using tensor::Tensor;
    using tensor::broadcast_to;

    std::cout << "[Test] broadcast_to" << std::endl;

    // 예제 1: shape [3, 1] → [3, 4]
    Tensor<float> x({3, 1}, {1.0f, 2.0f, 3.0f});
    Tensor<float> y = broadcast_to(x, {3, 4});

    assert(y.get_shape() == std::vector<size_t>({3, 4}));
    const auto& data = y.raw_data();

    // 각 row 값이 반복되는지 확인
    assert(data[y.get_strides()[0] * 0 + 0] == 1.0f);
    assert(data[y.get_strides()[0] * 1 + 0] == 2.0f);
    assert(data[y.get_strides()[0] * 2 + 0] == 3.0f);

    // row 전체가 동일한 값인지 (broadcast 확인)
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            assert(y({i, j}) == x({i, 0}));  // broadcasting된 값 비교

    // 예제 2: shape [1, 4] → [3, 4]
    Tensor<float> z({1, 4}, {10, 20, 30, 40});
    Tensor<float> z_b = broadcast_to(z, {3, 4});
    assert(z_b.get_shape() == std::vector<size_t>({3, 4}));

    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            assert(z_b({i, j}) == z({0, j}));  // row broadcasting 확인

    // 예제 3: 예외 테스트 (호환 불가능한 shape)
    bool caught = false;
    try {
        broadcast_to(x, {2, 4});  // [3,1] → [2,4]는 불가
    } catch (std::runtime_error& e) {
        caught = true;
    }
    assert(caught == true);

    std::cout << "✅ broadcast_to test passed.\n" << std::endl;
}

void test_sum_to() {
	std::cout << "[Test] sum_to\n";

    // 1. [2, 3, 4] → [2, 1, 4]
    Tensor<float> x({2, 3, 4}, {
         1,  2,  3,  4,
         5,  6,  7,  8,
         9, 10, 11, 12,

        13, 14, 15, 16,
        17, 18, 19, 20,
        21, 22, 23, 24
    });

    Tensor<float> y = sum_to(x, {2, 1, 4});
    assert(y.get_shape() == std::vector<size_t>({2, 1, 4}));

    // 각 행마다 3개의 row를 합친 결과 확인
    std::vector<float> expected = {
        15, 18, 21, 24,   // 1+5+9, 2+6+10, ...
        51, 54, 57, 60    // 13+17+21, ...
    };

    const auto& y_data = y.raw_data();
    for (size_t i = 0; i < expected.size(); ++i) {
        assert(std::abs(y_data[i] - expected[i]) < 1e-5);
    }

    // 2. [2, 3, 4] → [1, 3, 1]
    Tensor<float> z = sum_to(x, {1, 3, 1});
    assert(z.get_shape() == std::vector<size_t>({1, 3, 1}));

    std::vector<float> expected_z = {
        68,  100, 132  // column-wise sum across dim 0 and dim 2
    };

    const auto& z_data = z.raw_data();
    for (size_t i = 0; i < 3; ++i) {
        assert(std::abs(z_data[i] - expected_z[i]) < 1e-5);
    }

    // 3. 예외 케이스: 호환되지 않는 target shape
    bool caught = false;
    try {
        sum_to(x, {3, 2});  // ❌ invalid shape
    } catch (std::runtime_error&) {
        caught = true;
    }
    assert(caught == true);

    std::cout << "✅ sum_to test passed.\n" << std::endl;
}

void test_add_at() {
	Tensor<float> gx({3, 4}, 0.0f);  // 3x4 zero
	std::vector<size_t> slices = {1};
	Tensor<float> gy({4}, {1.0f, 2.0f, 3.0f, 4.0f});

	std::cout << "[Before add_at]\n";
	gx.show();

	add_at(gx, slices, gy);

	std::cout << "\n[After add_at(gx[1] += gy)]\n";
	gx.show();

	// ✅ 검증
	auto& result = gx.raw_data();
	assert(result[4] == 1.0f);  // gx[1][0]
	assert(result[5] == 2.0f);  // gx[1][1]
	assert(result[6] == 3.0f);  // gx[1][2]
	assert(result[7] == 4.0f);  // gx[1][3]

	// 나머지는 여전히 0이어야 함
	assert(result[0] == 0.0f);
	assert(result[1] == 0.0f);
	assert(result[2] == 0.0f);
	assert(result[3] == 0.0f);
	assert(result[8] == 0.0f);
	assert(result[9] == 0.0f);
	assert(result[10] == 0.0f);
	assert(result[11] == 0.0f);

	std::cout << "\n✅ test_add_at passed.\n";
}

int main() {
	test_tensor_sum();
	test_broadcast_to();
	test_sum_to();
	test_add_at();
}
