#include "deepczero.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>

void test_pooling_forward_backward() {
    using namespace function;

    // 입력 텐서: N=1, C=1, H=W=4
    // 값: 1..16 (행우선 증가)
    Tensor<> x({1, 1, 4, 4});
    {
        float v = 1.0f;
        for (size_t h = 0; h < 4; ++h) {
            for (size_t w = 0; w < 4; ++w) {
                x({0, 0, h, w}) = v++;
            }
        }
    }
    Variable vx(x);

    // 풀링 파라미터: kernel=2x2, stride=2x2, pad=0
    std::pair<size_t,size_t> kernel = {2, 2};
    std::pair<size_t,size_t> stride = {2, 2};
    std::pair<size_t,size_t> pad    = {0, 0};

    // 1) Forward 검증: 기대 출력은 [[6,8],[14,16]]
    Variable vy = pooling(vx, kernel, stride, pad);
    const Tensor<>& y = vy.data();
	vx.show();
	vy.show();

    assert(y.get_shape() == std::vector<size_t>({1, 1, 2, 2}));
	std::cout << "[y shape test passed]" << std::endl;
    {
        // ravel 순서로 기대값 나열 (NCHW = 1x1x2x2)
        Tensor<> y_expected({1,1,2,2}, {
            6.0f, 8.0f,
            14.0f, 16.0f
        });
        assert(is_allclose(y, y_expected) && "Pooling forward output mismatch");
    }
	std::cout << "[y value test passed]" << std::endl;

    // 2) Backward 검증 #1: gy = 1 (모두 1)
    // 비겹침 윈도우에서 최대값 위치로만 1이 흘러감
	vy.backward();
    const Tensor<>& gx1 = vx.grad().data();

    assert(gx1.get_shape() == std::vector<size_t>({1, 1, 4, 4}) && "gx shape must be [1,1,4,4]");
	std::cout << "[gx shape test passed]" << std::endl;

    // 기대 gx: 2x2 블록의 우하단(윈도우 내 argmax=3 위치)에만 1
    // 전역 좌표로는 (1,1), (1,3), (3,1), (3,3)
    {
        Tensor<> zero({1,1,4,4});
        // 비교를 위해 zero에 기대 위치만 1로 세팅
        zero({0,0,1,1}) = 1.0f;
        zero({0,0,1,3}) = 1.0f;
        zero({0,0,3,1}) = 1.0f;
        zero({0,0,3,3}) = 1.0f;

        // gx1 == zero 확인
        Tensor<> diff = gx1 - zero;
        // 모두 0이어야 함
        //Tensor<> diff_abs = diff.abs();
        // 합이 0인지 간단 체크
        float sum = 0.0f;
        Tensor<> flat = diff.ravel();
        for (size_t i = 0; i < flat.size(); ++i) sum += flat({i}) > 0 ? static_cast<float>(flat({i})) : -1 * static_cast<float>(flat({i}));
        assert(std::fabs(sum) < 1e-6f && "Pooling backward (ones) mismatch");
    }

	vx.grad().show();

	std::cout << "[gx value test passed]" << std::endl;

    std::cout << "[OK] test_pooling_forward_backward passed.\n";
}
 
// Reference max pooling via im2col + max (the previous CPU implementation)
static Tensor<> reference_pooling(const Tensor<>& x,
                                  std::pair<size_t,size_t> kernel,
                                  std::pair<size_t,size_t> stride,
                                  std::pair<size_t,size_t> pad) {
    Tensor<> padded = x;
    if (pad.first > 0 || pad.second > 0)
        padded = x.pad({{0, 0}, {0, 0}, {pad.first, pad.first}, {pad.second, pad.second}},
                       -std::numeric_limits<float>::infinity());
    Tensor<> col = tensor::im2col_array(padded, kernel, stride, {0, 0}, false);
    auto cs = col.get_shape();
    col = col.reshape({cs[0], cs[1], cs[2] * cs[3], cs[4], cs[5]});
    return col.max({2});
}

static Tensor<> random_tensor(std::vector<size_t> shape, unsigned seed) {
    Tensor<> t(shape);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : t.raw_data()) v = dist(gen);
    return t;
}

// Padding / stride / overlapping windows, compared against the im2col reference
void test_pooling_padding_stride() {
    struct Case {
        const char* name;
        std::vector<size_t> shape;
        std::pair<size_t,size_t> kernel, stride, pad;
    };
    const Case cases[] = {
        {"3x3 s1 p1",        {1, 4, 7, 7},     {3, 3}, {1, 1}, {1, 1}},
        {"5x5 s1 p2 (SPPF)", {1, 16, 20, 20},  {5, 5}, {1, 1}, {2, 2}},
        {"3x2 s2x1 p1x0",    {2, 2, 9, 6},     {3, 2}, {2, 1}, {1, 0}},
        {"2x2 s2 odd size",  {1, 1, 5, 5},     {2, 2}, {2, 2}, {0, 0}},
    };

    for (const auto& c : cases) {
        Tensor<> x = random_tensor(c.shape, 7);
        Tensor<> y_ref = reference_pooling(x, c.kernel, c.stride, c.pad);

        // forward (grad enabled)
        Variable vx(x);
        Variable vy = pooling(vx, c.kernel, c.stride, c.pad);
        assert(vy.data().get_shape() == y_ref.get_shape() && "Pooling output shape mismatch");
        assert(is_allclose(vy.data(), y_ref, 1e-6f, 1e-6f) && "Pooling forward mismatch vs reference");

        // forward (no grad): must give the same values without tracking indexes
        {
            dcz::UsingConfig no_grad("enable_backprop", false);
            Tensor<> y_ng = pooling(Variable(x), c.kernel, c.stride, c.pad).data();
            assert(is_allclose(y_ng, y_ref, 1e-6f, 1e-6f) && "Pooling no-grad forward mismatch");
        }

        // backward: every gy element is routed to exactly one input position,
        // so sum(gx) == sum(gy) and gx is zero where x is not a window max.
        Tensor<> gy = random_tensor(vy.shape(), 11);
        Variable vgy(gy);
        vy.set_grad(vgy);
        vy.backward();
        const Tensor<>& gx = vx.grad().data();
        assert(gx.get_shape() == x.get_shape() && "Pooling gx shape mismatch");

        float sum_gx = 0.0f, sum_gy = 0.0f;
        for (float v : gx.raw_data()) sum_gx += v;
        for (float v : gy.raw_data()) sum_gy += v;
        assert(std::fabs(sum_gx - sum_gy) < 1e-4f && "Pooling backward: gradient mass not preserved");

        std::cout << "[OK] pooling " << c.name << std::endl;
    }

    std::cout << "[OK] test_pooling_padding_stride passed.\n";
}

int main() {
    test_pooling_forward_backward();
    test_pooling_padding_stride();
    return 0;
}
