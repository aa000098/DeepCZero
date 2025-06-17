#include "deepczero.hpp"
#include <iostream>
#include <cassert>

using namespace tensor;

void test_broadcast_to_op() {
    std::cout << "[Test] Broadcast_To forward/backward\n";

    Variable x(Tensor<float>({3, 1}, {1.0f, 2.0f, 3.0f}));
    Variable y = broadcast_to(x, {3, 4});
    assert(y.shape() == std::vector<size_t>({3, 4}));

    // backward: should reduce to original shape [3, 1]
    y.backward();  // assume initial grad is ones

    auto gx = x.grad().data();
    const auto& g = gx.raw_data();

    // 각 row의 gradient는 4번 더해져야 함
    assert(g.size() == 3);
    assert(g[0] == 4.0f);
    assert(g[1] == 4.0f);
    assert(g[2] == 4.0f);

    std::cout << "✅ Broadcast_To test passed.\n" << std::endl;
}

void test_sum_to_op() {
    std::cout << "[Test] Sum_To forward/backward\n";

    Variable x(Tensor<float>({3, 4}, {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9,10,11,12
    }));

    Variable y = sum_to(x, {3, 1});
    assert(y.shape() == std::vector<size_t>({3, 1}));

    const auto& yd = y.data().raw_data();
    assert(yd[0] == 10);  // 1+2+3+4
    assert(yd[1] == 26);  // 5+6+7+8
    assert(yd[2] == 42);  // 9+10+11+12

    // backward: broadcast back to [3,4]
    y.backward();
    const auto& gx = x.grad();
	const auto& shape = gx.shape();
    assert(gx.size() == 12);
	assert(shape == std::vector<size_t>({3,4}));

	for (size_t i = 0; i < shape[0]; i++)
		for (size_t j = 0; j < shape[1]; j++)
			assert(std::abs(gx.data()({i, j}) - 1.0f) < 1e-5);  // broadcasted ones

    std::cout << "✅ Sum_To test passed.\n" << std::endl;
}

int main() {
    test_broadcast_to_op();
    test_sum_to_op();
    return 0;
}
