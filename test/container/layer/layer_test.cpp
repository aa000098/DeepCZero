#include "deepczero.hpp"

#include <cassert>
#include <iostream>
#include <cmath>

double pi = std::acos(-1.0f);

using namespace layer;

void test_layer_register_and_get_param() {
    // 1. 파라미터 생성 및 값 할당
    Parameter param({1.0f, 2.0f, 3.0f});

    // 2. Layer 객체 생성 및 등록
	Linear layer;
    layer.register_params("weight", param);

    // 3. 등록된 파라미터 조회 및 출력
    try {
        Parameter retrieved = layer.get_param("weight");
        std::cout << "[TEST] Parameter 'weight' found:\n";
        retrieved.data().show();  // 내부 텐서 출력
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception occurred: " << e.what() << std::endl;
    }

    // 4. 존재하지 않는 파라미터 조회 시도
    try {
        layer.get_param("bias");  // 존재하지 않음
    } catch (const std::exception& e) {
        std::cerr << "[EXPECTED ERROR] " << e.what() << std::endl;
    }
}

void test_linear_layer() {
    using namespace layer;

    std::cout << "[Test] Linear Layer Forward (with fixed weights)\n";

    // (1) 입력 텐서 (2 x 3)
    Tensor<> input_data({2, 3}, {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    });
    Variable x(input_data);

    // (2) Linear 레이어 생성
    Linear linear(2);  // input dim: 3, output dim: 2

    // (3) 고정된 weight와 bias 설정
    Tensor<> W_data({3, 2}, {
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 1.0f
    });
    Tensor<> b_data({2}, {1.0f, 2.0f});
	
	linear.get_param("W").data() = W_data;
	linear.get_param("b").data() = b_data;

    // (4) Forward 실행
    std::vector<Variable> inputs = {x};
    Variable y = linear(inputs);
	y.show();	

    // (5) 출력 검증
    const Tensor<>& y_data = y.data();
    const std::vector<float>& actual = y_data.raw_data();
    const std::vector<float> expected = {
        // y = x @ W + b
        // [1,2,3] @ [[1,0],[0,1],[1,1]] + [1,2]
        // = [1 + 0 + 3, 0 + 2 + 3] + [1,2] = [4,5] + [1,2] = [5,7]
        // [4,5,6] @ ... = [4+0+6, 0+5+6] + [1,2] = [10,11] + [1,2] = [11,13]
        5.0f, 7.0f,
        11.0f, 13.0f
    };

    assert(y_data.get_shape() == std::vector<size_t>({2, 2}));
    for (size_t i = 0; i < actual.size(); ++i) {
        assert(std::abs(actual[i] - expected[i]) < 1e-5);
    }

    std::cout << "✅ Linear layer test passed.\n";
}

Variable predict(Variable& x, Layer& l1, Layer& l2) {
	Variable y = l1(x);
	y = sigmoid(y);
	y = l2(y);
	return y;
}

void test_linear_regression() {

    std::cout << "[Test] Linear Regression Forward\n";

	Tensor x_data = rand(10, 1);
	Variable x = Variable(x_data, "x");
	Tensor noise_data = rand(10, 1);
	Variable noise = Variable(noise_data, "noise");
	Variable y = sin(2 * pi * x) + noise;

	Linear l1 = Linear(10);
	Linear l2 = Linear(1);

	float lr = 0.2;
	size_t iters = 10000;
	
	Variable y_pred;
	Variable loss;

	for (size_t i = 0; i < iters; i++) {
		y_pred = predict(x, l1, l2);
		loss = mean_squared_error(y, y_pred);
		loss.set_name("loss");

		l1.cleargrads();
		l2.cleargrads();
		loss.backward();

		
		for (auto* l : {&l1, &l2}) {
			for (auto& p : l->get_params())
				p.data() -= lr * p.grad().data();
		} 

		if (i % 1000 == 0) 
			loss.show();
	}


    std::cout << "✅ Linear Regression test passed.\n";

}

void test_linear_layer_3d() {
	using namespace layer;

	std::cout << "[Test] Linear Layer Forward (3D input, with flatten)\n";

	// (1) 입력 텐서: shape = [2, 2, 3]
	// 배치 2, 시퀀스 2, feature 3
	Tensor<> input_data({2, 2, 3}, {
		1.0f, 2.0f, 3.0f,  // sample 0-0
		4.0f, 5.0f, 6.0f,  // sample 0-1
		7.0f, 8.0f, 9.0f,  // sample 1-0
		10.0f, 11.0f, 12.0f  // sample 1-1
	});
	Variable x(input_data);

	// (2) Linear 레이어 (input_dim=3, output_dim=2)
	Linear linear(2);

	// (3) 고정된 weight와 bias 설정
	Tensor<> W_data({3, 2}, {
		1.0f, 0.0f,
		0.0f, 1.0f,
		1.0f, 1.0f
	});
	Tensor<> b_data({2}, {1.0f, 2.0f});
	linear.get_param("W").data() = W_data;
	linear.get_param("b").data() = b_data;

	// (4) Forward
	Variable y = linear({x});

	// (5) 출력 검증
	const Tensor<>& y_data = y.data();
	const std::vector<float>& actual = y_data.raw_data();
	const std::vector<float> expected = {
		// [1,2,3] @ W + b = [4,5] + [1,2] = [5,7]
		5.0f, 7.0f,
		11.0f, 13.0f,  // [4,5,6]
		17.0f, 19.0f,  // [7,8,9]
		23.0f, 25.0f   // [10,11,12]
	};

	assert(y_data.get_shape() == std::vector<size_t>({2, 2, 2}));
	for (size_t i = 0; i < actual.size(); ++i) {
		assert(std::abs(actual[i] - expected[i]) < 1e-5);
	}

	std::cout << "✅ 3D Linear layer test passed.\n";
}


int main() {
    test_layer_register_and_get_param();
	test_linear_layer();
	test_linear_regression();
	test_linear_layer_3d();
    return 0;
}

