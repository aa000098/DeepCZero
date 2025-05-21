#include "deepczero.hpp"

#include <cassert>
#include <iostream>

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

void test_tensor_dot_batched() {
    std::cout << "[Test] Tensor dot (batched)" << std::endl;

    using tensor::Tensor;

    // A: shape [2, 2, 2]
    Tensor<float> A({2, 2, 2}, {
        1, 2,
        3, 4,
        5, 6,
        7, 8
    });

    // B: shape [2, 2, 2]
    Tensor<float> B({2, 2, 2}, {
        1, 0,
        0, 1,
        2, 0,
        0, 2
    });

    // Expected result: shape [2, 2, 2]
    Tensor<float> expected({2, 2, 2}, {
        1, 2,
        3, 4,
        10, 12,
        14, 16
    });

    Tensor<float> C = dot(A, B);
    C.show();
	
	const auto& shape = C.get_shape();
    float eps = 1e-5;

    assert(C.get_shape() == expected.get_shape());

    size_t total = C.size();
    for (size_t flat = 0; flat < total; ++flat) {
        std::vector<size_t> idx(shape.size());
        size_t remaining = flat;
        for (int i = (int)shape.size() - 1; i >= 0; --i) {
            idx[i] = remaining % shape[i];
            remaining /= shape[i];
        }

        float actual = C(idx);
        float expect = expected(idx);
        assert(std::abs(actual - expect) < eps);
    }

    std::cout << "✅ dot (batched) test passed.\n" << std::endl;
}

void test_tensor_dot_4d() {
    std::cout << "[Test] Tensor dot (4D)" << std::endl;

    using tensor::Tensor;

    // A: shape [2, 1, 2, 3]
    Tensor<float> A({2, 1, 2, 3}, {
        // batch 0
        1, 2, 3,
        4, 5, 6,
        // batch 1
        7, 8, 9,
        10,11,12
    });

    // B: shape [2, 1, 3, 2]
    Tensor<float> B({2, 1, 3, 2}, {
        // batch 0
        1, 2,
        3, 4,
        5, 6,
        // batch 1
        6, 5,
        4, 3,
        2, 1
    });

    // Expected result: shape [2, 1, 2, 2]
    Tensor<float> expected({2, 1, 2, 2}, {
        // batch 0
        22, 28, 49, 64,
        // batch 1
        92, 68, 128, 95
    });

    Tensor<float> C = dot(A, B);
	C.show();
    
	const auto& shape = C.get_shape();
    float eps = 1e-5;

	assert(C.get_shape() == expected.get_shape());

    size_t total = C.size();
    for (size_t flat = 0; flat < total; ++flat) {
        std::vector<size_t> idx(shape.size());
        size_t remaining = flat;
        for (int i = (int)shape.size() - 1; i >= 0; --i) {
            idx[i] = remaining % shape[i];
            remaining /= shape[i];
        }

        float actual = C(idx);
        float expect = expected(idx);
        assert(std::abs(actual - expect) < eps);
    }

    std::cout << "✅ dot (4D) test passed.\n" << std::endl;
}

int main() {
    test_tensor_arithmetic();
	test_tensor_dot_batched();
	test_tensor_dot_4d();
    return 0;
}
