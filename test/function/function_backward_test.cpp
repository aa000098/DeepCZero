#include "deepczero.hpp"

#include <cassert>
#include <iostream>
#include <cmath>

using namespace function;

void test_function_forward_backward() {
    // 기본 입력
    Variable x({2.0f});
    Variable y({3.0f});
    Variable s({4.0f});  // scalar용


    // Square
    {
        auto f = std::make_shared<Square>();
        Variable out = f->forward({x});
        assert(out.data().raw_data()[0] == 4.0f);

        f->operator()({x}).backward();
        float grad = x.grad().data().raw_data()[0];
        assert(grad == 4.0f);  // 2 * x = 4
        x.cleargrad();
    }

    // Exp
    {
        auto f = std::make_shared<Exp>();
        Variable out = f->forward({x});
        float expected = std::exp(2.0f);
        assert(std::abs(out.data().raw_data()[0] - expected) < 1e-5f);

        f->operator()({x}).backward();
        float grad = x.grad().data().raw_data()[0];
        assert(std::abs(grad - expected) < 1e-5f);
        x.cleargrad();
    }

    // Add
    {
        auto f = std::make_shared<Add>();
        Variable out = f->forward({x, y});
        assert(out.data().raw_data()[0] == 5.0f);

        f->operator()({x, y}).backward();
        float gx = x.grad().data().raw_data()[0];
        float gy = y.grad().data().raw_data()[0];
        assert(gx == 1.0f && gy == 1.0f);
        x.cleargrad(); y.cleargrad();
    }

    // Mul
    {
        auto f = std::make_shared<Mul>();
        Variable out = f->forward({x, y});
        assert(out.data().raw_data()[0] == 6.0f);

        f->operator()({x, y}).backward();
        float gx = x.grad().data().raw_data()[0];
        float gy = y.grad().data().raw_data()[0];
        assert(gx == 3.0f && gy == 2.0f);
        x.cleargrad(); y.cleargrad();
    }

    // Neg
    {
        auto f = std::make_shared<Neg>();
        Variable out = f->forward({x});
        assert(out.data().raw_data()[0] == -2.0f);

        f->operator()({x}).backward();
        float gx = x.grad().data().raw_data()[0];
        assert(gx == -1.0f);
        x.cleargrad();
    }

    // Sub
    {
        auto f = std::make_shared<Sub>();
        Variable out = f->forward({x, y});
        assert(out.data().raw_data()[0] == -1.0f);

        f->operator()({x, y}).backward();
        float gx = x.grad().data().raw_data()[0];
        float gy = y.grad().data().raw_data()[0];
        assert(gx == 1.0f && gy == -1.0f);
        x.cleargrad(); y.cleargrad();
    }

    // Div
    {
        auto f = std::make_shared<Div>();
        Variable out = f->forward({y, x});
        assert(out.data().raw_data()[0] == 1.5f);  // 3 / 2

        f->operator()({y, x}).backward();
        float gx = y.grad().data().raw_data()[0];  // ∂L/∂y = 1/2
        float gy = x.grad().data().raw_data()[0];  // ∂L/∂x = -3/4
        assert(std::abs(gx - 0.5f) < 1e-5f);
        assert(std::abs(gy + 0.75f) < 1e-5f);
        x.cleargrad(); y.cleargrad();
    }

    // Pow
    {
        auto f = std::make_shared<Pow>();
        Variable out = f->forward({x, s});  // 2^4 = 16
        assert(out.data().raw_data()[0] == 16.0f);

        f->operator()({x, s}).backward();
        float gx = x.grad().data().raw_data()[0];  // 4 * 2^3 = 32
        assert(std::abs(gx - 32.0f) < 1e-5f);
        x.cleargrad(); s.cleargrad();
    }

    // Sin
    {
        auto f = std::make_shared<Sin>();
        Variable out = f->forward({x});
        float expected = std::sin(2.0f);
        assert(std::abs(out.data().raw_data()[0] - expected) < 1e-5f);

        f->operator()({x}).backward();
        float gx = x.grad().data().raw_data()[0];
        float expected_grad = std::cos(2.0f);
        assert(std::abs(gx - expected_grad) < 1e-5f);
        x.cleargrad();
    }

    // Cos
    {
        auto f = std::make_shared<Cos>();
        Variable out = f->forward({x});
        float expected = std::cos(2.0f);
        assert(std::abs(out.data().raw_data()[0] - expected) < 1e-5f);

        f->operator()({x}).backward();
        float gx = x.grad().data().raw_data()[0];
        float expected_grad = -std::sin(2.0f);
        assert(std::abs(gx - expected_grad) < 1e-5f);
        x.cleargrad();
    }

// Tanh
	{
		auto f = std::make_shared<Tanh>();
		Variable out = f->forward({x});
		float expected = std::tanh(2.0f);
		assert(std::abs(out.data().raw_data()[0] - expected) < 1e-5f);

		f->operator()({x}).backward();
		float gx = x.grad().data().raw_data()[0];
		float expected_grad = 1 - std::pow(std::tanh(2.0f), 2);
		assert(std::abs(gx - expected_grad) < 1e-5f);
		x.cleargrad();
	}

    // Reshape
    {
        Tensor t({2, 3});
        for (int i = 0; i < 6; ++i)
            t.raw_data()[i] = static_cast<float>(i + 1);  // [[1,2,3],[4,5,6]]
        Variable x_reshape(t);

        auto f = std::make_shared<Reshape>(std::vector<size_t>{3, 2});
        Variable out = f->forward({x_reshape});

        // Check shape
        std::vector<size_t> expected_shape = {3, 2};
        assert(out.data().get_shape() == expected_shape);

        // Check values: reshape preserves order
        std::vector<float> expected_data = {1, 2, 3, 4, 5, 6};
        for (size_t i = 0; i < expected_data.size(); ++i)
            assert(std::abs(out.data().raw_data()[i] - expected_data[i]) < 1e-5f);

        f->operator()({x_reshape}).backward();

        // Gradient should be all ones (reshaped back to original)
        std::vector<float> expected_grad(6, 1.0f);
        for (size_t i = 0; i < expected_grad.size(); ++i)
            assert(std::abs(x_reshape.grad().data().raw_data()[i] - expected_grad[i]) < 1e-5f);

        x_reshape.cleargrad();
    }

    // Transpose
    {
        Tensor t({2, 3, 4, 5});
        for (size_t i = 0; i < t.size(); ++i)
            t.raw_data()[i] = static_cast<float>(i + 1);  // [[1,2,3],[4,5,6]]
        Variable x_trans(t);

        auto f = std::make_shared<Transpose>(std::vector<size_t>{1, 0, 3, 2});
        Variable out = f->forward({x_trans});
        
        // Check shape
        std::vector<size_t> expected_shape = {3, 2, 5, 4};
        assert(out.data().get_shape() == expected_shape);

        f->operator()({x_trans}).backward();

        // Gradient should be all ones (transposed back)
        std::vector<size_t> expected_grad_shape = {2, 3, 4, 5};
        assert(x_trans.grad().shape() == expected_grad_shape);

        x_trans.cleargrad();
    }

    std::cout << "[✓] All Function forward/backward tests passed." << std::endl;
}

int main() {
    test_function_forward_backward();
    return 0;
}
