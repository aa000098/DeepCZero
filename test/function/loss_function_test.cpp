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
	std::cout << "[Test] Softmax Cross Entropy Error Forward Backward" << std::endl;

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

void test_softmax_cross_entropy_error_forward_2() {
    using namespace function;
	std::cout << "[Test] Softmax Cross Entropy Error Test2" << std::endl;

    // 입력: x (logits), t (정답 인덱스)
    Tensor<> x_data({4, 2}, {
        0.2f, -0.4f, 0.3f, 0.5f,
        1.3f, -3.2f, 2.1f, 0.3f
    });
    Tensor<> t_data({4}, {1, 0, 1, 0});  // 정답 인덱스

    Variable x(x_data);
    Variable t(t_data);


    // Forward
    Variable loss = softmax_cross_entropy_error(x, t);

    // 🔍 정답 계산 (수기 계산)
    float expected_loss = 1.6249;

    float actual_loss = loss.data().raw_data()[0];
    assert(std::abs(actual_loss - expected_loss) < 1e-4f);

    std::cout << "[✅ Forward passed] loss = " << actual_loss << std::endl;

}

// === Abs Tests ===
void test_abs_forward_backward() {
	std::cout << "[Test] Abs forward/backward" << std::endl;

	Tensor<> t({4}, {-2.0f, -1.0f, 0.0f, 3.0f});
	Variable x(t);

	Variable y = abs(x);
	// expected: [2, 1, 0, 3]
	assert(std::abs(y.data()({0}) - 2.0f) < 1e-5);
	assert(std::abs(y.data()({1}) - 1.0f) < 1e-5);
	assert(std::abs(y.data()({2}) - 0.0f) < 1e-5);
	assert(std::abs(y.data()({3}) - 3.0f) < 1e-5);

	y.backward();
	// sign: [-1, -1, 0, 1]
	assert(std::abs(x.grad().data()({0}) - (-1.0f)) < 1e-5);
	assert(std::abs(x.grad().data()({1}) - (-1.0f)) < 1e-5);
	assert(std::abs(x.grad().data()({2}) - 0.0f) < 1e-5);
	assert(std::abs(x.grad().data()({3}) - 1.0f) < 1e-5);

	std::cout << "  Abs test passed" << std::endl;
}

// === Clamp Tests ===
void test_clamp_forward_backward() {
	std::cout << "[Test] Clamp forward/backward" << std::endl;

	Tensor<> t({5}, {-3.0f, -0.5f, 0.5f, 1.5f, 3.0f});
	Variable x(t);

	Variable y = clamp(x, -1.0f, 2.0f);
	// expected: [-1, -0.5, 0.5, 1.5, 2]
	assert(std::abs(y.data()({0}) - (-1.0f)) < 1e-5);
	assert(std::abs(y.data()({1}) - (-0.5f)) < 1e-5);
	assert(std::abs(y.data()({2}) - 0.5f) < 1e-5);
	assert(std::abs(y.data()({3}) - 1.5f) < 1e-5);
	assert(std::abs(y.data()({4}) - 2.0f) < 1e-5);

	Variable loss = sum(y);
	loss.backward();
	// mask: [0, 1, 1, 1, 0] (clamped at boundaries → grad 0)
	assert(std::abs(x.grad().data()({0}) - 0.0f) < 1e-5);
	assert(std::abs(x.grad().data()({1}) - 1.0f) < 1e-5);
	assert(std::abs(x.grad().data()({2}) - 1.0f) < 1e-5);
	assert(std::abs(x.grad().data()({3}) - 1.0f) < 1e-5);
	assert(std::abs(x.grad().data()({4}) - 0.0f) < 1e-5);

	std::cout << "  Clamp test passed" << std::endl;
}

// === BinaryCrossEntropy Tests ===
void test_binary_cross_entropy_forward() {
	std::cout << "[Test] BinaryCrossEntropy forward" << std::endl;

	// PyTorch reference:
	// x = torch.tensor([0.5, -0.3, 1.2, -0.8])
	// t = torch.tensor([1.0, 0.0, 1.0, 0.0])
	// F.binary_cross_entropy_with_logits(x, t) = 0.5765
	Tensor<> x_data({4}, {0.5f, -0.3f, 1.2f, -0.8f});
	Tensor<> t_data({4}, {1.0f, 0.0f, 1.0f, 0.0f});

	Variable x(x_data);
	Variable t(t_data);

	Variable loss = binary_cross_entropy(x, t);
	float actual = loss.data()({0});

	// Manual: mean of [max(0.5,0)-0.5*1+log(1+exp(-0.5)),
	//                   max(-0.3,0)-(-0.3)*0+log(1+exp(-0.3)),
	//                   max(1.2,0)-1.2*1+log(1+exp(-1.2)),
	//                   max(-0.8,0)-(-0.8)*0+log(1+exp(-0.8))]
	// = mean([0.4741, 0.5544, 0.2634, 0.3711]) = 0.4157
	// Actually let me compute properly:
	// x=0.5: max(0.5,0) - 0.5 + log(1+exp(-0.5)) = 0.5 - 0.5 + 0.4741 = 0.4741
	// x=-0.3: max(-0.3,0) - 0 + log(1+exp(-0.3)) = 0 + 0 + 0.5544 = 0.5544  (wait, -(-0.3)*0=0)
	// Actually: max(x,0) - x*t + log(1+exp(-|x|))
	// x=0.5,t=1: 0.5-0.5+log(1+exp(-0.5)) = 0+0.4741 = 0.4741
	// x=-0.3,t=0: 0-0+log(1+exp(-0.3)) = 0.5544
	// x=1.2,t=1: 1.2-1.2+log(1+exp(-1.2)) = 0+0.2634 = 0.2634
	// x=-0.8,t=0: 0-0+log(1+exp(-0.8)) = 0.3711
	// mean = (0.4741+0.5544+0.2634+0.3711)/4 = 0.4157
	float expected = 0.4157f;
	std::cout << "  loss = " << actual << " (expected ~" << expected << ")" << std::endl;
	assert(std::abs(actual - expected) < 1e-3f);

	std::cout << "  BinaryCrossEntropy forward test passed" << std::endl;
}

void test_binary_cross_entropy_backward() {
	std::cout << "[Test] BinaryCrossEntropy backward" << std::endl;

	Tensor<> x_data({4}, {0.5f, -0.3f, 1.2f, -0.8f});
	Tensor<> t_data({4}, {1.0f, 0.0f, 1.0f, 0.0f});

	Variable x(x_data);
	Variable t(t_data);

	Variable loss = binary_cross_entropy(x, t);
	loss.backward();

	// grad = (sigmoid(x) - t) / N
	// sigmoid(0.5)=0.6225, sigmoid(-0.3)=0.4256, sigmoid(1.2)=0.7685, sigmoid(-0.8)=0.3100
	// grads = (0.6225-1)/4, (0.4256-0)/4, (0.7685-1)/4, (0.3100-0)/4
	//       = -0.0944, 0.1064, -0.0579, 0.0775
	float expected_grads[] = {-0.0944f, 0.1064f, -0.0579f, 0.0775f};
	for (size_t i = 0; i < 4; i++) {
		float g = x.grad().data()({i});
		assert(std::abs(g - expected_grads[i]) < 1e-3f);
	}

	std::cout << "  BinaryCrossEntropy backward test passed" << std::endl;
}

int main() {
	test_mean_squared_error_forward();
	test_mean_squared_error_backward();
	std::cout << "✅ All MeanSquaredError tests passed." << std::endl;

	test_softmax_cross_entropy_error_forward_backward();
	test_softmax_cross_entropy_error_forward_2();

	test_abs_forward_backward();
	test_clamp_forward_backward();
	test_binary_cross_entropy_forward();
	test_binary_cross_entropy_backward();
	std::cout << "✅ All new loss function tests passed." << std::endl;

	return 0;
}

