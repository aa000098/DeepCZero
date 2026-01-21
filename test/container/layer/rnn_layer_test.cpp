#include "deepczero.hpp"

void test_rnn_layer() {
    std::cout << "=== Test: RNN Layer Forward Pass ===" << std::endl;

    // RNN 레이어 생성
    size_t hidden_size = 10;
    layer::RNN rnn(hidden_size);
    std::cout << "1. RNN layer created with hidden size " << hidden_size << std::endl;

    // 임의의 입력 데이터 생성 (배치 크기 2, 시퀀스 길이 5)
    size_t seq_length = 1;
    std::vector<Variable> inputs;
    for (size_t t = 0; t < seq_length; ++t) {
        Tensor<> x({1, 1});
        x = rand({1, 1});
        inputs.push_back(Variable(x));
    }
    std::cout << "2. Created random input sequence of length " << seq_length 
              << " with batch size " << 1 << std::endl;
    // RNN 레이어에 입력 데이터 전달하여 순전파 수행
    Variable h = rnn.forward(inputs);
    std::cout << "3. Forward pass completed." << std::endl;

    // 출력 형태 확인
    std::cout << "4. Output shape: ";
    for (auto dim : h.shape()) {
        std::cout << dim << " ";
    }
    std::cout << std::endl;
}

int main() {
    test_rnn_layer();
    return 0;
}