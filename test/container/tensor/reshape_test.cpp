#include "container/tensor/tensor.hpp"  // Tensor<T> and TensorND<T>
#include <iostream>
#include <cassert>

using namespace tensor;

void test_tensor_reshape() {
    std::cout << "[Test] Tensor reshape" << std::endl;

    // 1. Create a 2x2 TensorND
    Tensor<float> a = Tensor<float>({2, 2}, std::vector<float>{1, 2, 3, 4});
    assert(a.get_shape() == std::vector<size_t>({2, 2}));
    assert(a.raw_data()[0] == 1);
    assert(a.raw_data()[3] == 4);

    // 2. Reshape to (4,)
    Tensor<float> b = a.reshape({4});
    assert(b.get_shape() == std::vector<size_t>({4}));

    // 3. Check that data is shared (view)
    assert(&a.raw_data()[0] == &b.raw_data()[0]);  // Same data_ptr
    assert(b.raw_data()[0] == 1.0f);
    assert(b.raw_data()[1] == 2.0f);
    assert(b.raw_data()[2] == 3.0f);
    assert(b.raw_data()[3] == 4.0f);

    // 4. Modify b and check a is affected (view test)
    b.raw_data()[0] = 10.0f;
    assert(a.raw_data()[0] == 10.0f);

    std::cout << "✅ reshape test passed.\n" << std::endl;
}

int main() {
    test_tensor_reshape();
    return 0;
}

