#include "deepczero.hpp"

#include <cassert>

using namespace std;

void test_tensor_slice() {
    // (4, 3) 텐서 생성
    Tensor<float> t({4, 3});
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            t({i, j}) = static_cast<float>(i * 10 + j);
        }
    }

    // dim=0, start=1, end=3 → 슬라이스하면 shape=(2, 3)
    Tensor sliced = t.slice(0, 1, 3);

    // shape 확인
    auto shape = sliced.get_shape();
    assert(shape.size() == 2);
    assert(shape[0] == 2 && shape[1] == 3);

    // 값 확인
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            float expected = static_cast<float>((i + 1) * 10 + j);
            assert(sliced({i, j}) == expected);
        }
    }

    cout << "[PASSED] Tensor slice test" << endl;
}

int main() {
    test_tensor_slice();
    return 0;
}
