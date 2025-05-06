#include "deepczero.hpp"

#include <iostream>
#include <cassert>

void test_variable_ops() {
    Variable a({2.0f});
    Variable b({3.0f});
    float s = 5.0f;

    // Add
    assert((a + b).data().raw_data()[0] == 5.0f);
    assert((a + s).data().raw_data()[0] == 7.0f);
    assert((s + b).data().raw_data()[0] == 8.0f);

    // Sub
    assert((b - a).data().raw_data()[0] == 1.0f);
    assert((s - a).data().raw_data()[0] == 3.0f);
    assert((a - s).data().raw_data()[0] == -3.0f);

    // Mul
    assert((a * b).data().raw_data()[0] == 6.0f);
    assert((a * s).data().raw_data()[0] == 10.0f);
    assert((s * b).data().raw_data()[0] == 15.0f);

    // Div
    assert((b / a).data().raw_data()[0] == 1.5f);
    assert((a / s).data().raw_data()[0] == 0.4f);

    // Pow
    assert((a ^ 3.0f).data().raw_data()[0] == 8.0f);

    // Neg
    assert((-a).data().raw_data()[0] == -2.0f);

    std::cout << "[✓] All Variable op tests passed." << std::endl;
}

int main() {
    test_variable_ops();
    return 0;
}
