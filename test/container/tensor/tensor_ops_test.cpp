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

void test_tensor_tensordot_basic() {
    // A: shape (2, 3)
    Tensor<float> A({2, 3}, {1,2,3,4,5,6});

    // B: shape (3, 4)
    Tensor<float> B({3, 4}, {7,8,9,10,11,12,13,14,15,16,17,18});
    
	// Compute tensordot over axis 1 of A and axis 0 of B → result shape (2, 4)
    Tensor<float> Y = tensordot(A, B, {{1}, {0}});

    // Check output shape
    const auto& shape = Y.get_shape();
    assert(shape.size() == 2);
    assert(shape[0] == 2 && shape[1] == 4);

    // Expected result manually calculated:
    // Y[0] = [1*7 + 2*11 + 3*15, ..., 1*10 + 2*14 + 3*18]
    // Y[1] = [4*7 + 5*11 + 6*15, ..., 4*10 + 5*14 + 6*18]
    Tensor expected(
		{2, 4}, 
		{ 74, 80, 86, 92, 173, 188, 203, 218 });

	Y.show();
    assert(is_allclose(Y, expected));

    std::cout << "✅ test_tensordot_basic passed.\n";
}

void test_tensor_tensordot_highdim() {
    // A: shape (2, 3, 2, 2)
    Tensor<float> A(
        {2, 3, 2, 2},
        {
            1, 2, 3, 4,     // [0,0,:,:]
            5, 6, 7, 8,     // [0,1,:,:]
            9, 10, 11, 12,  // [0,2,:,:]
            13, 14, 15, 16, // [1,0,:,:]
            17, 18, 19, 20, // [1,1,:,:]
            21, 22, 23, 24  // [1,2,:,:]
        }
    );

    // B: shape (4, 3, 2, 2, 2)
    Tensor<float> B(
        {4, 3, 2, 2, 2},
        {
            1,2,3,4,5,6,7,8,9,10,11,12,
            13,14,15,16,17,18,19,20,21,22,23,24,
            25,26,27,28,29,30,31,32,33,34,35,36,
            37,38,39,40,41,42,43,44,45,46,47,48,
            49,50,51,52,53,54,55,56,57,58,59,60,
            61,62,63,64,65,66,67,68,69,70,71,72,
            73,74,75,76,77,78,79,80,81,82,83,84,
            85,86,87,88,89,90,91,92,93,94,95,96,
//            97,98,99,100,101,102,103,104,105,106,107,108,
//            109,110,111,112,113,114,115,116,117,118,119,120,
//            121,122,123,124,125,126,127,128,129,130,131,132,
//            133,134,135,136,137,138,139,140,141,142,143,144
        }
    );

    // Expected result: shape (2, 4, 2)
    Tensor<float> expected(
        {2, 4, 2},
        {
            1222, 1300,
            3094, 3172,
            4966, 5044,
            6838, 6916,

            2950, 3172,
            8278, 8500,
            13606, 13828,
            18934, 19156
        }
    );

    // Perform tensordot
    Tensor<float> Y = tensordot(A, B, {{1,2,3}, {1,2,3}});

	A.show();
	B.show();
	Y.show();

    assert(is_allclose(Y, expected));
    std::cout << "✅ test_tensor_tensordot_small passed.\n";
}

int main() {
    test_tensor_arithmetic();
	test_tensor_dot_batched();
	test_tensor_dot_4d();
	test_tensor_tensordot_basic();
	test_tensor_tensordot_highdim();
    return 0;
}
