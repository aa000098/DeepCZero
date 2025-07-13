#include "deepczero.hpp"

#include <cassert>
#include <iostream>

void test_tensor_pad() {
    std::cout << "[Test] Tensor pad" << std::endl;

    // 1. Create a 2x2 Tensor
    Tensor<int> tensor({2, 2}, {1, 2, 3, 4});
    assert(tensor.get_shape() == std::vector<size_t>({2, 2}));

    // 2. Pad the tensor with zeros (1 padding each side)
    Tensor<int> padded_tensor = tensor.pad({{1,1}, {1,1}}, 0);
    assert(padded_tensor.get_shape() == std::vector<size_t>({4, 4}));

    // 3. Check padded values explicitly
    assert(padded_tensor({1,1}) == 1);
    assert(padded_tensor({1,2}) == 2);
    assert(padded_tensor({2,1}) == 3);
    assert(padded_tensor({2,2}) == 4);
    assert(padded_tensor({0,0}) == 0);  // padding area check
    assert(padded_tensor({3,3}) == 0);  // padding area check

    // 4. Modify padded_tensor and verify tensor is unaffected (not view)
    padded_tensor({1,1}) = 10;
    assert(tensor({0,0}) == 1);  // original tensor remains unchanged

    tensor.show();
    padded_tensor.show();

    std::cout << "✅ pad test passed.\n" << std::endl;
}

int main() {
	test_tensor_pad();
}
