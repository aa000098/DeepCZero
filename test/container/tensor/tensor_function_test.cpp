#include "deepczero.hpp"

#include <cassert>

void test_tensor_sum() {
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

    std::cout << "✅ tensor sum tests passed." << std::endl;
}

int main() {
	test_tensor_sum();
}
