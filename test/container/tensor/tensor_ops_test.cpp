#include "deepczero.hpp"

#include <cassert>
#include <iostream>

using tensor::Tensor;
using namespace tensor;

void test_tensor_arithmetic() {
    // 초기화: a = [1, 2, 3], b = [4, 5, 6]
    Tensor<float> a({3}, {1.0f, 2.0f, 3.0f});
    Tensor<float> b({3}, {4.0f, 5.0f, 6.0f});

    // 1. a += b
    Tensor<float> a1 = a.clone();
    a1 += b;
    assert(a1.raw_data() == std::vector<float>({5.0f, 7.0f, 9.0f}));

    // 2. c = a + b
    Tensor<float> c = a + b;
    assert(c.raw_data() == std::vector<float>({5.0f, 7.0f, 9.0f}));

    // 3. a * b (element-wise)
    Tensor<float> d = a * b;
    assert(d.raw_data() == std::vector<float>({4.0f, 10.0f, 18.0f}));

    // 4. a += 2.0f
    Tensor<float> a2 = a.clone();
    a2 += 2.0f;
    assert(a2.raw_data() == std::vector<float>({3.0f, 4.0f, 5.0f}));

    // 5. e = a + 2.0f
    Tensor<float> e = a + 2.0f;
    assert(e.raw_data() == std::vector<float>({3.0f, 4.0f, 5.0f}));

    // 6. f = 10.0f - a
    Tensor<float> f = 10.0f - a;
    assert(f.raw_data() == std::vector<float>({9.0f, 8.0f, 7.0f}));

    // 7. g = 2.0f + a
    Tensor<float> g = 2.0f + a;
    assert(g.raw_data() == std::vector<float>({3.0f, 4.0f, 5.0f}));

    // 8. a *= 3.0f
    Tensor<float> a3 = a.clone();
    a3 *= 3.0f;
    assert(a3.raw_data() == std::vector<float>({3.0f, 6.0f, 9.0f}));

    std::cout << "✅ Tensor arithmetic test passed!" << std::endl;
}

int main() {
    test_tensor_arithmetic();
    return 0;
}
