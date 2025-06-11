#include "deepczero.hpp"
#include <cassert>
#include <iostream>

void test_mean_squared_error_forward() {
	std::cout << "[Test] MeanSquaredError forward()" << std::endl;

	// x0 = [1, 2, 3], x1 = [2, 2, 2] → diff = [-1, 0, 1]
	Tensor<> t0({3}, {1, 2, 3});
	Tensor<> t1({3}, {2, 2, 2});

	Variable x0(t0);
	Variable x1(t1);

	Variable loss = mean_squared_error(x0, x1);

	float expected = (1.0f + 0.0f + 1.0f) / 3.0f;  // = 0.666...
	assert(std::abs(loss.data()({0}) - expected) < 1e-5);
}

void test_mean_squared_error_backward() {
	std::cout << "[Test] MeanSquaredError backward()" << std::endl;

	Tensor<> t0({3}, {1, 2, 3});
	Tensor<> t1({3}, {2, 2, 2});

	Variable x0(t0);
	Variable x1(t1);

	Variable loss = mean_squared_error(x0, x1);  // wrapper function or use Function directly
	loss.backward();

	// expected gradient: dL/dx0 = (2/N) * (x0 - x1)
	std::vector<float> expected_grad = {-0.6666f, 0.0f, 0.6666f};

	for (size_t i = 0; i < 3; ++i) {
		float g = x0.grad().data()({i});
		assert(std::abs(g - expected_grad[i]) < 1e-3);
	}
}

void test_softmax_cross_entropy_error_forward_backward() {
    using namespace function;

    // 입력: x (logits), t (정답 인덱스)
    Tensor<> x_data({2, 3}, {
        1.0f, 2.0f, 3.0f,
        1.0f, 2.0f, 3.0f
    });
    Tensor<> t_data({2}, {2, 0});  // 정답 인덱스

    Variable x(x_data);
    Variable t(t_data);

    // Forward
    Variable loss = softmax_cross_entropy_error(x, t);

    // 🔍 정답 계산 (수기 계산)
    // softmax([1,2,3]) = [0.0900306, 0.244728, 0.665241]
    float logp1 = std::log(0.665241f);  // label 2
    float logp2 = std::log(0.0900306f); // label 0
    float expected_loss = -(logp1 + logp2) / 2.0f;

    float actual_loss = loss.data().raw_data()[0];
    assert(std::abs(actual_loss - expected_loss) < 1e-4f);

    std::cout << "[✅ Forward passed] loss = " << actual_loss << std::endl;

    // Backward
    loss.backward();

    const auto& grad = x.grad().data().raw_data();

    // 정답 그래디언트: (softmax - one_hot) / N
    // softmax: [0.0900306, 0.244728, 0.665241]
    float expected_grad[6] = {
        0.0450153f, 0.122364f, -0.16738f,  // label=2
        -0.454985f, 0.122364f, 0.332620f   // label=0
    };

    for (int i = 0; i < 6; ++i) {
        assert(std::abs(grad[i] - expected_grad[i]) < 1e-4f);
    }

    std::cout << "[✅ Backward passed] x.grad = ";
    x.grad().show();

    std::cout << "\n🎉 SoftmaxCrossEntropyError forward/backward test passed.\n";
}

int main() {
	test_mean_squared_error_forward();
	test_mean_squared_error_backward();
	std::cout << "✅ All MeanSquaredError tests passed." << std::endl;

	test_softmax_cross_entropy_error_forward_backward();

	return 0;
}

