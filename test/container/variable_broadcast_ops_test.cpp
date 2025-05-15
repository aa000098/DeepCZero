#include "deepczero.hpp"  // Variable, Add, Sub, Mul, Div 포함
#include <cassert>
#include <iostream>
#include <cmath>

void test_variable_broadcast_ops() {
    std::cout << "[Test] Variable broadcast ops (forward + backward)" << std::endl;

    const float eps = 1e-5;

    // x: shape (3, 4)
    Variable x(Tensor<float>({3, 4}, {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9,10,11,12
    }));

    // y: shape (1, 4) - broadcast along first dimension
    Variable y(Tensor<float>({1, 4}, {1, 1, 1, 1}));

    // ADD
    Variable z_add = x + y;  // shape (3, 4)
    z_add.backward();
	//z_add.show();
	//x.grad().show();
	//y.grad().show();

    // check grad of x
    for (float g : x.grad().data().raw_data())
        assert(std::abs(g - 1.0f) < eps);

    // check grad of y: shape (1, 4) → broadcast된 축 합산 → grad = [3, 3, 3, 3]
    const auto& gy = y.grad().data().raw_data();
    for (float g : gy)
        assert(std::abs(g - 3.0f) < eps);

    x.cleargrad(); y.cleargrad();

    // SUB
    Variable z_sub = x - y;
    z_sub.backward();
    for (float g : x.grad().data().raw_data())
        assert(std::abs(g - 1.0f) < eps);
    for (float g : y.grad().data().raw_data())
        assert(std::abs(g + 3.0f) < eps);  // -gy, so should be -3

    x.cleargrad(); y.cleargrad();

    // MUL
    Variable z_mul = x * y;
    z_mul.backward();
//	z_mul.show();
//	x.grad().show();
//	y.grad().show();
    for (size_t i = 0; i < x.grad().size(); ++i)
        assert(std::abs(x.grad().data().raw_data()[i] - 1.0f) < eps);  // y = 1, so grad = x * 1
    for (size_t i = 0; i < y.grad().size(); i++)
        assert(std::abs(y.grad().data()({0,i}) - x.data().sum({0})({i})) < eps);  // row sum of x column-wise (shape (1,4))

    x.cleargrad(); y.cleargrad();

    // DIV
    Variable z_div = x / y;
    z_div.backward();
    for (size_t i = 0; i < x.grad().size(); ++i)
        assert(std::abs(x.grad().data().raw_data()[i] - 1.0f) < eps);  // 1/y = 1
    for (size_t i = 0; i < y.grad().size(); ++i) {
        float expected = -(x.data().raw_data()[i + 0] +
                           x.data().raw_data()[i + 4] +
                           x.data().raw_data()[i + 8]) / (1.0f * 1.0f);  // -x / y^2
        assert(std::abs(y.grad().data().raw_data()[i] - expected) < eps);
    }

    std::cout << "✅ Variable broadcast ops test passed.\n" << std::endl;
}

int main() {
    test_variable_broadcast_ops();
    return 0;
}

