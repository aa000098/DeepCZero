#include "container/tensor/tensor_all.hpp" 

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

	a.show();
	b.show();
    std::cout << "✅ reshape test passed.\n" << std::endl;
}

void test_tensor_transpose() {
    std::cout << "[Test] Tensor transpose" << std::endl;

    // 1. Create a 2x3 Tensor
    Tensor<float> a({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
    assert(a.get_shape() == std::vector<size_t>({2, 3}));

    // 2. Default transpose → reverse axes = [1, 0]
    Tensor<float> b = a.transpose();
    assert(b.get_shape() == std::vector<size_t>({3, 2}));
    assert(b.raw_data()[0] == 1);  // view test: same data
    assert(&a.raw_data()[0] == &b.raw_data()[0]);

    // 3. Specified axes transpose = {1, 0}
    Tensor<float> c = a.transpose({1, 0});
    assert(c.get_shape() == std::vector<size_t>({3, 2}));
    assert(&a.raw_data()[0] == &c.raw_data()[0]);

    // 4. Value check: check indices if indexing supported (optional)

    a.show();
    b.show();
    c.show();
    std::cout << "✅ transpose test passed.\n" << std::endl;
}

int main() {
    test_tensor_reshape();
    test_tensor_transpose();
    return 0;
}

