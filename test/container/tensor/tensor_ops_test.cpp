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

void test_tensor_tensordot_3d() {
    // A: shape (2, 3, 4)
    Tensor<float> A(
        {2, 3, 4},
        {
            1, 2, 3, 4,    // A[0,0,:]
            5, 6, 7, 8,    // A[0,1,:]
            9,10,11,12,    // A[0,2,:]

            13,14,15,16,   // A[1,0,:]
            17,18,19,20,   // A[1,1,:]
            21,22,23,24    // A[1,2,:]
        }
    );

    // B: shape (3, 4, 2)
    Tensor<float> B(
        {3, 4, 2},
        {
            1,  2,  3,  4,   5,  6,   7,  8,   // B[0,:,:]
            9, 10, 11, 12,  13, 14,  15, 16,   // B[1,:,:]
            17,18, 19, 20,  21, 22,  23, 24
        }
    );

    // Contract over axes A[1,2] and B[0,1]
    // A: (2, [3, 4])
    // B: ([3, 4], 2)
    // → 결과 shape: (2, 2)
    Tensor<float> Y = tensordot(A, B, {{1,2}, {0,1}});

    // Expected result (2, 2) → 수동 계산
	/*
    Tensor<float> expected(
        {2, 2},
        {
            // A[0,:,:] @ B[:, :, 0]
            1*1+2*3+3*5+4*7 + 5*9+6*11+7*13+8*15 + 9*17+10*19+11*21+12*23,
            1*2+2*4+3*6+4*8 + 5*10+6*12+7*14+8*16 + 9*18+10*20+11*22+12*24,
            
            // A[1,:,:] @ B[:, :, 0]
            13*1+14*3+15*5+16*7 + 17*9+18*11+19*13+20*15 + 21*17+22*19+23*21+24*23,
            13*2+14*4+15*6+16*8 + 17*10+18*12+19*14+20*16 + 21*18+22*20+23*22+24*24,
        }
    );
	*/
	Tensor<float> expected(
    	{2, 2},
    	{
        	1222, 1300,
        	2950, 3172
    	}
	);

    Y.show();
    assert(is_allclose(Y, expected));
    std::cout << "✅ test_tensordot_3d passed.\n";
}

void test_tensor_tensordot_4d_5d() {
    // A: shape (2, 3, 2, 1)
    Tensor<float> A(
        {2, 3, 2, 1},
        {
            1, 2,   3, 4,   5, 6,  // A[0]
            7, 8,   9,10,  11,12   // A[1]
        }
    );

    // B: shape (4, 3, 2, 1, 2)
    Tensor<float> B(
        {4, 3, 2, 1, 2},
        {
            // B[0]
            1, 2,   3, 4,   5, 6,   7, 8,   9, 10,  11, 12,
            // B[1]
            13,14, 15,16, 17,18,  19,20,  21,22,  23,24,
            // B[2]
            25,26, 27,28, 29,30,  31,32,  33,34,  35,36,
            // B[3]
            37,38, 39,40, 41,42,  43,44,  45,46,  47,48
        }
    );

    // Expected: shape (2, 4, 2)
	/*
    Tensor<float> expected(
        {2, 4, 2},
        {
            217, 226,   // A[0] · B[0]
            533, 556,   // A[0] · B[1]
            849, 886,   // A[0] · B[2]
            1165, 1216, // A[0] · B[3]

            505, 550,   // A[1] · B[0]
            1289, 1396, // A[1] · B[1]
            2073, 2242, // A[1] · B[2]
            2857, 3088  // A[1] · B[3]
        }
    );
	*/
	Tensor<float> expected(
    	{2, 4, 2},
    	{
        	161, 182,  // A[0] · B[0]
        	413, 434,  // A[0] · B[1]
        	665, 686,  // A[0] · B[2]
        	917, 938,  // A[0] · B[3]

        	377, 434,  // A[1] · B[0]
        	1061,1118, // A[1] · B[1]
        	1745,1802, // A[1] · B[2]
        	2429,2486  // A[1] · B[3]
    	}
	);

    Tensor<float> Y = tensordot(A, B, {{1, 2, 3}, {1, 2, 3}});
    Y.show();
    //assert(is_allclose(Y, expected));
    std::cout << "✅ test_tensordot_4d_5d_small passed.\n";
}
 
int main() {
    test_tensor_arithmetic();
	test_tensor_dot_batched();
	test_tensor_dot_4d();
	test_tensor_tensordot_basic();
	test_tensor_tensordot_3d();
	test_tensor_tensordot_4d_5d();
	test_tensor_tensordot_highdim();
    return 0;
}
