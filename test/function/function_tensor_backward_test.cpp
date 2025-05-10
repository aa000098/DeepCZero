#include "deepczero.hpp"
#include <cassert>
#include <iostream>
#include <cmath>

void test_function_forward_backward() {
    Variable x({2.0f, 3.0f});
    Variable y({3.0f, 4.0f});
    Variable s({4.0f});  // scalar
    const float eps = 1e-5f;

    // Square
    {
        auto f = std::make_shared<Square>();
        Variable out = f->forward({x});
        assert(std::abs(out.data().raw_data()[0] - 4.0f) < eps);
        assert(std::abs(out.data().raw_data()[1] - 9.0f) < eps);

        f->operator()({x}).backward();
        assert(std::abs(x.grad().data().raw_data()[0] - 4.0f) < eps); // 2*2
        assert(std::abs(x.grad().data().raw_data()[1] - 6.0f) < eps); // 2*3
        x.cleargrad();
    }

    // Exp
    {
        auto f = std::make_shared<Exp>();
        Variable out = f->forward({x});
        for (int i = 0; i < 2; ++i)
            assert(std::abs(out.data().raw_data()[i] - std::exp(x.data().raw_data()[i])) < eps);

        f->operator()({x}).backward();
        for (int i = 0; i < 2; ++i)
            assert(std::abs(x.grad().data().raw_data()[i] - std::exp(x.data().raw_data()[i])) < eps);
        x.cleargrad();
    }

    // Add
    {
        auto f = std::make_shared<Add>();
        Variable out = f->forward({x, y});
        for (int i = 0; i < 2; ++i)
            assert(std::abs(out.data().raw_data()[i] - (x.data().raw_data()[i] + y.data().raw_data()[i])) < eps);

        f->operator()({x, y}).backward();
        for (int i = 0; i < 2; ++i) {
            assert(std::abs(x.grad().data().raw_data()[i] - 1.0f) < eps);
            assert(std::abs(y.grad().data().raw_data()[i] - 1.0f) < eps);
        }
        x.cleargrad(); y.cleargrad();
    }

    // Mul
    {
        auto f = std::make_shared<Mul>();
        Variable out = f->forward({x, y});
        for (int i = 0; i < 2; ++i)
            assert(std::abs(out.data().raw_data()[i] - (x.data().raw_data()[i] * y.data().raw_data()[i])) < eps);

        f->operator()({x, y}).backward();
        for (int i = 0; i < 2; ++i) {
            assert(std::abs(x.grad().data().raw_data()[i] - y.data().raw_data()[i]) < eps);
            assert(std::abs(y.grad().data().raw_data()[i] - x.data().raw_data()[i]) < eps);
        }
        x.cleargrad(); y.cleargrad();
    }

    // Neg
    {
        auto f = std::make_shared<Neg>();
        Variable out = f->forward({x});
        for (int i = 0; i < 2; ++i)
            assert(std::abs(out.data().raw_data()[i] + x.data().raw_data()[i]) < eps);

        f->operator()({x}).backward();
        for (int i = 0; i < 2; ++i)
            assert(std::abs(x.grad().data().raw_data()[i] + 1.0f) < eps);
        x.cleargrad();
    }

    // Sub
    {
        auto f = std::make_shared<Sub>();
        Variable out = f->forward({x, y});
        for (int i = 0; i < 2; ++i)
            assert(std::abs(out.data().raw_data()[i] - (x.data().raw_data()[i] - y.data().raw_data()[i])) < eps);

        f->operator()({x, y}).backward();
        for (int i = 0; i < 2; ++i) {
            assert(std::abs(x.grad().data().raw_data()[i] - 1.0f) < eps);
            assert(std::abs(y.grad().data().raw_data()[i] + 1.0f) < eps);
        }
        x.cleargrad(); y.cleargrad();
    }

    // Div
    {
        auto f = std::make_shared<Div>();
        Variable out = f->forward({x, s});
   		float scalar = s.data().raw_data()[0];
        for (int i = 0; i < 2; ++i)
            assert(std::abs(out.data().raw_data()[i] - (x.data().raw_data()[i] / scalar)) < eps);

        f->operator()({x, s}).backward();

        //float expected_s_grad = 0.0f;
        for (int i = 0; i < 2; ++i) {
            float gx_expected = 1.0f / scalar;
       //     float gs_expected = -x.data().raw_data()[i] / (scalar * scalar);
            assert(std::abs(x.grad().data().raw_data()[i] - gx_expected) < eps);
       //     expected_s_grad += gs_expected;
        }
		
        //assert(std::abs(s.grad().data().raw_data()[0] - expected_s_grad) < eps);

        x.cleargrad(); s.cleargrad();
    }

    // Pow (with scalar)
    {
        auto f = std::make_shared<Pow>();
        Variable out = f->forward({x, s});
        for (int i = 0; i < 2; ++i)
            assert(std::abs(out.data().raw_data()[i] - std::pow(x.data().raw_data()[i], s.data().raw_data()[0])) < eps);

        f->operator()({x, s}).backward();
        for (int i = 0; i < 2; ++i) {
            float base = x.data().raw_data()[i];
            float exp = s.data().raw_data()[0];
            float gx = exp * std::pow(base, exp - 1);
            assert(std::abs(x.grad().data().raw_data()[i] - gx) < eps);
        }
        x.cleargrad(); s.cleargrad();
    }

    // Sin
    {
        auto f = std::make_shared<Sin>();
        Variable out = f->forward({x});
        for (int i = 0; i < 2; ++i)
            assert(std::abs(out.data().raw_data()[i] - std::sin(x.data().raw_data()[i])) < eps);

        f->operator()({x}).backward();
        for (int i = 0; i < 2; ++i)
            assert(std::abs(x.grad().data().raw_data()[i] - std::cos(x.data().raw_data()[i])) < eps);
        x.cleargrad();
    }

    // Cos
    {
        auto f = std::make_shared<Cos>();
        Variable out = f->forward({x});
        for (int i = 0; i < 2; ++i)
            assert(std::abs(out.data().raw_data()[i] - std::cos(x.data().raw_data()[i])) < eps);

        f->operator()({x}).backward();
        for (int i = 0; i < 2; ++i)
            assert(std::abs(x.grad().data().raw_data()[i] + std::sin(x.data().raw_data()[i])) < eps);
        x.cleargrad();
    }

    // Tanh
    {
        auto f = std::make_shared<Tanh>();
        Variable out = f->forward({x});
        for (int i = 0; i < 2; ++i)
            assert(std::abs(out.data().raw_data()[i] - std::tanh(x.data().raw_data()[i])) < eps);

        f->operator()({x}).backward();
        for (int i = 0; i < 2; ++i) {
            float tanh_val = std::tanh(x.data().raw_data()[i]);
            float expected_grad = 1 - tanh_val * tanh_val;
            assert(std::abs(x.grad().data().raw_data()[i] - expected_grad) < eps);
        }
        x.cleargrad();
    }

    std::cout << "[✓] All Function forward/backward (all elements) tests passed." << std::endl;
}

int main() {
    test_function_forward_backward();
    return 0;
}

