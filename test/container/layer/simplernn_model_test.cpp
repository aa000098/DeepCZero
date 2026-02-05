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

    for (size_t i = 0; i < seqlen - 1; ++i) {  // -1 because dataset_t is shorter
        // BPTT에서는 reset_state를 호출하지 않음 (hidden state 유지)
        // Python DeZero처럼 epoch 시작 시에만 reset

        Variable x = dataset_x[i];
        Variable y = simplernn({x});
        Variable t = dataset_t[i];

        Variable curr_loss = mean_squared_error(y, t);

        // 첫 번째 iteration이거나 loss가 reset된 경우
        if (count == 0) {
            loss = curr_loss;
        } else {
            loss = loss + curr_loss;
        }
        count++;

        // bptt_length마다 또는 마지막에 backward
        if (count == bptt_length || i == seqlen - 2) {
            float avg_loss = loss.data().raw_data()[0] / count;
            std::cout << "   Step " << i << " - Avg Loss: " << avg_loss << std::endl;

            simplernn.cleargrads();
            loss.backward();
            loss.unchain_backward();  // graph 끊기

            count = 0;
            loss = Variable();  // Reset loss
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
    simplernn.reset_state();  // epoch 시작시 한번만 reset
    Variable loss;

    for (size_t i = 0; i < seqlen - 1; ++i) {
        Variable x = dataset_x[i];
        Variable y = simplernn({x});
        Variable t = dataset_t[i];
        Variable curr_loss = mean_squared_error(y, t);

        // loss 누적
        if (count == 0) {
            loss = curr_loss;
        } else {
            loss = loss + curr_loss;
        }
        count++;

        // bptt_length마다 backward & update
        if (count == bptt_length || i == seqlen - 2) {
            float avg_loss = loss.data().raw_data()[0] / count;
            total_loss += avg_loss;
            loss_count++;

            // 매 update마다 평균 loss 출력 (accumulated loss를 사용)
            if (loss_count % 10 == 0) {
                float running_avg = total_loss / loss_count;
                std::cout << "   Update " << loss_count << " (Step " << i << ") - Avg Loss: " << avg_loss
                          << ", Running Avg: " << running_avg << std::endl;
            }

            simplernn.cleargrads();
            loss.backward();
            loss.unchain_backward();  // graph 끊기
            optimizer.update();

            count = 0;
            loss = Variable();  // Reset loss
        }
    }
    std::cout << "4. Training completed!" << std::endl;

    // 예측 테스트
    std::cout << "5. Testing predictions..." << std::endl;
    simplernn.reset_state();
    float test_error = 0;
    float warmup_error = 0;  // 처음 10 스텝 (warmup)
    float stable_error = 0;   // 나머지 스텝
    size_t test_samples = 50;
    size_t warmup_steps = 10;

    for (size_t i = 0; i < test_samples; ++i) {
        Variable x = dataset_x[i];
        Variable y = simplernn({x});
        Variable t = dataset_t[i];

        float pred = y.data().raw_data()[0];
        float target = t.data().raw_data()[0];
        float error = std::abs(pred - target);
        test_error += error;

        if (i < warmup_steps) {
            warmup_error += error;
        } else {
            stable_error += error;
        }

        if (i % 5 == 0) {  // 더 자주 출력
            std::cout << "   Step " << i << " - Pred: " << pred
                      << ", Target: " << target << ", Error: " << error << std::endl;
        }
    }

    std::cout << "\n=== Test Results ===" << std::endl;
    std::cout << "   Warmup error (steps 0-9): " << (warmup_error / warmup_steps) << std::endl;
    std::cout << "   Stable error (steps 10+): " << (stable_error / (test_samples - warmup_steps)) << std::endl;

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

    std::cout << "Testing full BPTT with two windows..." << std::endl;
    simplernn.reset_state();
    simplernn.cleargrads();

    // First BPTT window (steps 0-1)
    Variable y0 = simplernn({xs[0]});
    Variable l0 = mean_squared_error(y0, ts[0]);
    Variable y1 = simplernn({xs[1]});
    Variable l1 = mean_squared_error(y1, ts[1]);
    Variable loss1 = l0 + l1;
    std::cout << "Window 1 (steps 0-1): Loss = " << loss1.data().raw_data()[0] << std::endl;

    simplernn.cleargrads();
    loss1.backward();
    // optimizer.update();  // TEMPORARILY DISABLED
    std::cout << "Window 1 backward completed (no update)!" << std::endl;

    // Clean up window 1
    loss1.unchain_backward();
    simplernn.cleargrads();
    y0 = Variable();
    y1 = Variable();
    l0 = Variable();
    l1 = Variable();
    loss1 = Variable();

    // Second BPTT window (steps 2-3)
    Variable y2 = simplernn({xs[2]});
    Variable l2 = mean_squared_error(y2, ts[2]);
    Variable y3 = simplernn({xs[3]});
    Variable l3 = mean_squared_error(y3, ts[3]);
    Variable loss2 = l2 + l3;
    std::cout << "Window 2 (steps 2-3): Loss = " << loss2.data().raw_data()[0] << std::endl;

    simplernn.cleargrads();
    std::cout << "About to call backward 2..." << std::endl;
    loss2.backward(false, false, false);  // debug=false
    std::cout << "Backward 2 completed! About to call optimizer.update()..." << std::endl;
    optimizer.update();
    std::cout << "Window 2 backward & update completed!" << std::endl;

    std::cout << "\n=== BPTT test completed successfully! ===" << std::endl;

    std::cout << "Test completed successfully!" << std::endl;
}

int main() {
    test_simplernn_forward_only();
    test_simplernn_single_backward();
    test_simplernn_model();
    test_bptt_simple();
    test_simplrnn_sin_wave_prediction();
    return 0;
}
         