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
        dataset_x.push_back(Variable(x));
        if (i != 0) 
            dataset_t.push_back(Variable(x));
    }
    std::cout << "2. Created dataset with " << seqlen << " samples" << std::endl;

    std::cout << "3. Starting training with BPTT (using recursive unchain_backward)..." << std::endl;

    size_t bptt_length = 2;  // Testing with length 2
    simplernn.reset_state();
    Variable loss;
    size_t count = 0;

    for (size_t i = 0; i < seqlen; ++i) {
        // BPTT에서는 reset_state를 호출하지 않음 (hidden state 유지)
        // Python DeZero처럼 epoch 시작 시에만 reset

        std::vector<Variable> x_vec = {dataset_x[i]};
        Variable x = dataset_x[i];
        Variable y = simplernn({x});
        Variable t = dataset_t[i];


        loss += mean_squared_error(y, t);
        count++;

        // bptt_length마다 또는 마지막에 backward
        if (count == bptt_length || i == seqlen - 1) {
            float avg_loss = loss.data().raw_data()[0] / count;
            std::cout << "   Step " << i << " - Avg Loss: " << avg_loss << std::endl;

            simplernn.cleargrads();
            loss.backward();
            // loss.backward(false, false, true);  // debug mode disabled for speed
            // loss.unchain_backward();  // 재귀적으로 graph 끊기

            count = 0;
        }
    }

    std::cout << "4. Training completed!" << std::endl;

}

void test_simplrnn_sin_wave_prediction() {
    std::cout << "\n=== Test: SimpleRNN Sine Wave Prediction ===" << std::endl;

    size_t hidden_size = 10;
    size_t output_size = 1;
    float learning_rate = 0.01;
    SimpleRNN simplernn(hidden_size, output_size);
    SGD optimizer(learning_rate);
    optimizer.setup(simplernn);
    std::cout << "1. SimpleRNN created" << std::endl;

    simplernn.reset_state();

    // Sine wave 데이터 생성
    size_t seqlen = 500;
    size_t bptt_length = 5;  // Reduce to avoid deep graph
    float total_loss = 0;
    size_t loss_count = 0;
    size_t count = 0;
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
        
        // bptt_length마다 backward & update
        if (count == bptt_length || i == seqlen - 2) {
            float avg_loss = loss.data().raw_data()[0] / count;
            total_loss += avg_loss;
            loss_count++;

            simplernn.cleargrads();
            loss.backward();
            loss.unchain_backward();  // graph 끊기
            optimizer.update();

            count = 0;
        }

        simplernn.cleargrads();
        loss.backward();
        loss.unchain_backward();
    }
    std::cout << "4. Training completed!" << std::endl;

    // 예측 테스트
    std::cout << "5. Testing predictions..." << std::endl;
    simplernn.reset_state();
    float test_error = 0;
    size_t test_samples = 50;

    for (size_t i = 0; i < test_samples; ++i) {
        Variable x = dataset_x[i];
        Variable y = simplernn({x});
        Variable t = dataset_t[i];

        float pred = y.data().raw_data()[0];
        float target = t.data().raw_data()[0];
        float error = std::abs(pred - target);
        test_error += error;

        if (i % 10 == 0) {
            std::cout << "   Step " << i << " - Pred: " << pred
                      << ", Target: " << target << ", Error: " << error << std::endl;
        }
    }

    std::cout << "6. Average test error: " << test_error / test_samples << std::endl;
}


void test_bptt_simple() {
    std::cout << "\n=== Test: Simple BPTT Debug ===" << std::endl;

    SimpleRNN simplernn(5, 1);
    SGD optimizer(0.1);
    optimizer.setup(simplernn);

    // 매우 간단한 데이터 (이름 포함)
    std::vector<Variable> xs, ts;
    for (int i = 0; i < 10; ++i) {
        Tensor<> x({1, 1}, float(i));
        Tensor<> t({1, 1}, float(i+1));
        xs.push_back(Variable(x, "x" + std::to_string(i)));
        ts.push_back(Variable(t, "t" + std::to_string(i)));
    }

    std::cout << "Starting 2 BPTT iterations with length=2..." << std::endl;
    simplernn.reset_state();

    // First BPTT window (steps 0-1)
    Variable y0 = simplernn({xs[0]});
    y0.set_name("y0");
    Variable l0 = mean_squared_error(y0, ts[0]);
    l0.set_name("loss0");

    Variable y1 = simplernn({xs[1]});
    y1.set_name("y1");
    Variable l1 = mean_squared_error(y1, ts[1]);
    l1.set_name("loss1_step");

    Variable loss1 = l0 + l1;
    loss1.set_name("loss1_total");
    std::cout << "Step 0-1: Loss = " << loss1.data().raw_data()[0] << std::endl;

    plot_dot_graph(loss1, true, "bptt_step01_before_backward.dot");
    std::cout << "Graph before backward saved to bptt_step01_before_backward.dot" << std::endl;

    simplernn.cleargrads();
    loss1.backward();
    std::cout << "Backward 1 completed" << std::endl;

    loss1.unchain_backward();
    std::cout << "loss1.unchain_backward() completed" << std::endl;

    simplernn.unchain_hidden();  // BPTT 구간 끝: hidden state의 creator 끊기
    std::cout << "simplernn.unchain_hidden() completed" << std::endl;

    plot_dot_graph(loss1, true, "bptt_step01_after_unchain.dot");
    std::cout << "Graph after unchain saved to bptt_step01_after_unchain.dot" << std::endl;

    optimizer.update();
    std::cout << "Update 1 completed" << std::endl;

    std::cout << "Manually clearing all intermediate variables..." << std::endl;
    // Step 0-1의 모든 intermediate variables 제거
    y0 = Variable();
    y1 = Variable();
    l0 = Variable();
    l1 = Variable();
    loss1 = Variable();

    // 모든 gradient를 완전히 제거
    for (auto& x : xs) x.cleargrad();
    for (auto& t : ts) t.cleargrad();

    // Parameters의 gradient creator도 끊기 - 중요!
    // gradient variable들도 computation graph를 가지고 있어서 이미 해제된 variable을 참조할 수 있음
    // Note: gradient가 존재하는지 확인할 방법이 없으므로 일단 clear만 호출
    simplernn.cleargrads();

    std::cout << "All variables cleared, now creating step 2-3..." << std::endl;

    // Second BPTT window (steps 2-3)
    Variable y2 = simplernn({xs[2]});
    y2.set_name("y2");
    Variable l2 = mean_squared_error(y2, ts[2]);
    l2.set_name("loss2_step");

    Variable y3 = simplernn({xs[3]});
    y3.set_name("y3");
    Variable l3 = mean_squared_error(y3, ts[3]);
    l3.set_name("loss3_step");

    Variable loss2 = l2 + l3;
    loss2.set_name("loss2_total");
    std::cout << "Step 2-3: Loss = " << loss2.data().raw_data()[0] << std::endl;

    plot_dot_graph(loss2, true, "bptt_step23_before_backward.dot");
    std::cout << "Graph before backward 2 saved to bptt_step23_before_backward.dot" << std::endl;

    std::cout << "About to call backward 2..." << std::endl;
    loss2.backward(false, false, true);  // debug=true to see where it crashes
    std::cout << "Backward 2 completed" << std::endl;

    optimizer.update();
    std::cout << "Update 2 completed" << std::endl;

    std::cout << "Test completed successfully!" << std::endl;
}

int main() {
    // test_simplernn_forward_only();
    // test_simplernn_single_backward();
    // test_simplernn_model();
    test_bptt_simple();
    // test_simplrnn_sin_wave_prediction();
    return 0;
}
    