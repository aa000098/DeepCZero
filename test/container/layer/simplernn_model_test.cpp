#include "deepczero.hpp"

void test_simplernn_forward_only() {
    std::cout << "=== Test: SimpleRNN Forward Only ===" << std::endl;

    size_t hidden_size = 10;
    size_t output_size = 1;
    SimpleRNN simplernn(hidden_size, output_size);
    std::cout << "1. SimpleRNN created" << std::endl;

    simplernn.reset_state();

    // 10개 샘플로 forward만 테스트
    for (size_t i = 0; i < 10; ++i) {
        Tensor<> x = randn({1, 1});
        std::vector<Variable> x_vec = {Variable(x)};
        Variable y = simplernn(x_vec);

        std::cout << "   Step " << i << " - Output: " << y.data().raw_data()[0] << std::endl;
    }

    std::cout << "2. Forward test completed!" << std::endl;
}

void test_simplernn_single_backward() {
    std::cout << "\n=== Test: SimpleRNN Single Backward ===" << std::endl;

    size_t hidden_size = 10;
    size_t output_size = 1;
    SimpleRNN simplernn(hidden_size, output_size);
    std::cout << "1. SimpleRNN created" << std::endl;

    simplernn.reset_state();

    // 1개 샘플로 backward 테스트
    Tensor<> x = randn({1, 1});
    Tensor<> t = randn({1, 1});

    std::vector<Variable> x_vec = {Variable(x)};
    Variable y = simplernn(x_vec);
    Variable loss = mean_squared_error(y, Variable(t));

    std::cout << "2. Loss: " << loss.data().raw_data()[0] << std::endl;

    simplernn.cleargrads();
    std::cout << "3. Running backward..." << std::endl;
    loss.backward();
    std::cout << "4. Backward completed!" << std::endl;
}

void test_simplernn_model() {
    std::cout << "=== Test: SimpleRNN Model Forward Pass ===" << std::endl;

    // SimpleRNN 모델 생성
    size_t hidden_size = 10;
    size_t output_size = 1;
    SimpleRNN simplernn(hidden_size, output_size);
    std::cout << "1. SimpleRNN model created with hidden_size=" << hidden_size
              << " output_size=" << output_size << std::endl;

    // 데이터셋 생성 (Python처럼 시퀀스 데이터)
    size_t seqlen = 100;
    size_t batch_size = 1;
    size_t input_size = 1;

    std::vector<Variable> dataset_x;
    std::vector<Variable> dataset_t;

    for (size_t i = 0; i < seqlen; ++i) {
        Tensor<> x = randn({batch_size, input_size});
        Tensor<> t = randn({batch_size, output_size});
        dataset_x.push_back(Variable(x));
        dataset_t.push_back(Variable(t));
    }
    std::cout << "2. Created dataset with " << seqlen << " samples" << std::endl;

    std::cout << "3. Starting training (independent steps, no BPTT)..." << std::endl;
    std::cout << "   Note: Each step resets state to avoid graph accumulation." << std::endl;
    std::cout << "   For proper BPTT, RNN layer needs unchain support." << std::endl;

    for (size_t i = 0; i < seqlen; ++i) {
        // Reset state to break computation graph
        // (proper BPTT would require RNN layer with unchain support)
        simplernn.reset_state();

        std::vector<Variable> x_vec = {dataset_x[i]};
        Variable y = simplernn(x_vec);
        Variable loss = mean_squared_error(y, dataset_t[i]);

        if (i % 10 == 0) {
            std::cout << "   Step " << i << " - Loss: " << loss.data().raw_data()[0] << std::endl;
        }

        simplernn.cleargrads();
        loss.backward();
    }

    std::cout << "4. Training completed!" << std::endl;

}

void test_simplrnn_sin_wave_prediction() {
    std::cout << "\n=== Test: SimpleRNN Sine Wave Prediction ===" << std::endl;

    size_t hidden_size = 10;
    size_t output_size = 1;
    SimpleRNN simplernn(hidden_size, output_size);
    std::cout << "1. SimpleRNN created" << std::endl;

    simplernn.reset_state();

    // Sine wave 데이터 생성
    size_t seqlen = 50;
    std::vector<Variable> dataset_x;
    std::vector<Variable> dataset_t;

    for (size_t i = 0; i < seqlen; ++i) {
        float val = std::sin(2 * 3.14159f * i / 25.0f);
        Tensor<> x({1, 1});
        x({0, 0}) = val;
        Tensor<> t({1, 1});
        t({0, 0}) = std::sin(2 * 3.14159f * (i + 1) / 25.0f); // 다음 값 예측
        dataset_x.push_back(Variable(x));
        dataset_t.push_back(Variable(t));
    }
    std::cout << "2. Created sine wave dataset" << std::endl;

    // 순전파 및 역전파 테스트
    std::cout << "3. Training on sine wave..." << std::endl;
    for (size_t i = 0; i < seqlen - 1; ++i) {
        // Reset state to avoid graph accumulation
        simplernn.reset_state();

        std::vector<Variable> x_vec = {dataset_x[i]};
        Variable y = simplernn(x_vec);
        Variable loss = mean_squared_error(y, dataset_t[i]);

        if (i % 10 == 0) {
            std::cout << "   Step " << i << " - Loss: " << loss.data().raw_data()[0] << std::endl;
        }

        simplernn.cleargrads();
        loss.backward();
    }

    std::cout << "4. Sine wave prediction test completed!" << std::endl;
}

int main() {
    test_simplernn_forward_only();
    test_simplernn_single_backward();
    test_simplernn_model();
    test_simplrnn_sin_wave_prediction();
    return 0;
}