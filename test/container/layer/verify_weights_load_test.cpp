#include "deepczero.hpp"

#include <iostream>
#include <cmath>

// 두 텐서가 동일한지 검증 (floating point tolerance 적용)
bool tensors_equal(const Tensor<>& a, const Tensor<>& b, float eps = 1e-6f) {
    if (a.get_shape() != b.get_shape()) {
        std::cout << "  [FAIL] Shape mismatch: ";
        for (auto s : a.get_shape()) std::cout << s << " ";
        std::cout << "vs ";
        for (auto s : b.get_shape()) std::cout << s << " ";
        std::cout << std::endl;
        return false;
    }

    const auto& data_a = a.contiguous().raw_data();
    const auto& data_b = b.contiguous().raw_data();

    for (size_t i = 0; i < data_a.size(); ++i) {
        if (std::abs(data_a[i] - data_b[i]) > eps) {
            std::cout << "  [FAIL] Value mismatch at index " << i
                      << ": " << data_a[i] << " vs " << data_b[i] << std::endl;
            return false;
        }
    }
    return true;
}

// MLP 모델의 save/load 검증
void test_mlp_save_load_verify() {
    std::cout << "=== Test: MLP Save/Load Verification ===" << std::endl;

    // 1) 원본 모델 생성 및 forward로 weight 초기화
    MLP model_original({10, 5, 3});

    Tensor<> x_data({2, 10}, 1.0f);  // batch=2, input=10
    Variable x(x_data);
    Variable y_original = model_original(x);

    // 2) 원본 weight 값들 저장
    auto original_params = model_original.flatten_params();
    std::unordered_map<std::string, Tensor<>> original_weights;
    for (const auto& [name, param] : original_params) {
        original_weights[name] = param.data().contiguous();
    }

    // 3) 파일로 저장
    std::string weight_path = "test_mlp_verify.dcz";
    model_original.save_weights(weight_path);
    std::cout << "  Saved weights to: " << weight_path << std::endl;

    // 4) 새 모델 생성 후 로드
    MLP model_loaded({10, 5, 3});
    // forward 한번 호출해서 weight shape 초기화
    Variable dummy = model_loaded(x);

    std::filesystem::path full_path = get_cache_file_path("weights/" + weight_path);
    model_loaded.load_weights(full_path.string());
    std::cout << "  Loaded weights from: " << full_path << std::endl;

    // 5) 로드된 weight와 원본 비교
    auto loaded_params = model_loaded.flatten_params();

    bool all_passed = true;
    for (const auto& [name, param] : loaded_params) {
        auto it = original_weights.find(name);
        if (it == original_weights.end()) {
            std::cout << "  [FAIL] Parameter '" << name << "' not found in original" << std::endl;
            all_passed = false;
            continue;
        }

        bool equal = tensors_equal(it->second, param.data());
        if (equal) {
            std::cout << "  [PASS] Parameter '" << name << "' matches" << std::endl;
        } else {
            all_passed = false;
        }
    }

    // 6) forward 결과도 동일한지 확인
    Variable y_loaded = model_loaded(x);

    bool forward_match = tensors_equal(y_original.data(), y_loaded.data());
    if (forward_match) {
        std::cout << "  [PASS] Forward output matches" << std::endl;
    } else {
        std::cout << "  [FAIL] Forward output mismatch" << std::endl;
        all_passed = false;
    }

    std::cout << (all_passed ? "=== MLP Test PASSED ===" : "=== MLP Test FAILED ===") << std::endl;
    std::cout << std::endl;
}

// Linear 단일 레이어 검증
void test_linear_save_load_verify() {
    std::cout << "=== Test: Linear Layer Save/Load Verification ===" << std::endl;

    // 1) 원본 레이어 생성
    layer::Linear linear_original(5);

    Tensor<> x_data({3, 10}, 0.5f);
    Variable x(x_data);
    Variable y_original = linear_original(x);

    // 2) 원본 weight 복사
    Tensor<> original_W = linear_original.get_param("W").data().contiguous();
    Tensor<> original_b = linear_original.get_param("b").data().contiguous();

    // 3) 저장
    std::string weight_path = "test_linear_verify.dcz";
    linear_original.save_weights(weight_path);
    std::cout << "  Saved weights" << std::endl;

    // 4) 새 레이어에 로드
    layer::Linear linear_loaded(5);
    Variable dummy = linear_loaded(x);  // shape 초기화

    std::filesystem::path full_path = get_cache_file_path("weights/" + weight_path);
    linear_loaded.load_weights(full_path.string());
    std::cout << "  Loaded weights" << std::endl;

    // 5) 비교
    bool w_match = tensors_equal(original_W, linear_loaded.get_param("W").data());
    bool b_match = tensors_equal(original_b, linear_loaded.get_param("b").data());

    std::cout << (w_match ? "  [PASS] W matches" : "  [FAIL] W mismatch") << std::endl;
    std::cout << (b_match ? "  [PASS] b matches" : "  [FAIL] b mismatch") << std::endl;

    // forward 결과 비교
    Variable y_loaded = linear_loaded(x);
    bool forward_match = tensors_equal(y_original.data(), y_loaded.data());
    std::cout << (forward_match ? "  [PASS] Forward output matches" : "  [FAIL] Forward output mismatch") << std::endl;

    bool all_passed = w_match && b_match && forward_match;
    std::cout << (all_passed ? "=== Linear Test PASSED ===" : "=== Linear Test FAILED ===") << std::endl;
    std::cout << std::endl;
}

// Conv2d 레이어 검증
void test_conv2d_save_load_verify() {
    std::cout << "=== Test: Conv2d Layer Save/Load Verification ===" << std::endl;

    // 1) 원본 레이어 생성
    layer::Conv2d conv_original(8, {3, 3}, {1, 1}, {1, 1}, false, 3);

    Tensor<> x_data({1, 3, 8, 8}, 0.5f);
    Variable x(x_data);
    Variable y_original = conv_original(x);

    // 2) 원본 weight 복사
    Tensor<> original_W = conv_original.get_param("W").data().contiguous();
    Tensor<> original_b = conv_original.get_param("b").data().contiguous();

    // 3) 저장
    std::string weight_path = "test_conv2d_verify.dcz";
    conv_original.save_weights(weight_path);
    std::cout << "  Saved weights" << std::endl;

    // 4) 새 레이어에 로드
    layer::Conv2d conv_loaded(8, {3, 3}, {1, 1}, {1, 1}, false, 3);
    Variable dummy = conv_loaded(x);  // shape 초기화

    std::filesystem::path full_path = get_cache_file_path("weights/" + weight_path);
    conv_loaded.load_weights(full_path.string());
    std::cout << "  Loaded weights" << std::endl;

    // 5) 비교
    bool w_match = tensors_equal(original_W, conv_loaded.get_param("W").data());
    bool b_match = tensors_equal(original_b, conv_loaded.get_param("b").data());

    std::cout << (w_match ? "  [PASS] W matches" : "  [FAIL] W mismatch") << std::endl;
    std::cout << (b_match ? "  [PASS] b matches" : "  [FAIL] b mismatch") << std::endl;

    // forward 결과 비교
    Variable y_loaded = conv_loaded(x);
    bool forward_match = tensors_equal(y_original.data(), y_loaded.data());
    std::cout << (forward_match ? "  [PASS] Forward output matches" : "  [FAIL] Forward output mismatch") << std::endl;

    bool all_passed = w_match && b_match && forward_match;
    std::cout << (all_passed ? "=== Conv2d Test PASSED ===" : "=== Conv2d Test FAILED ===") << std::endl;
    std::cout << std::endl;
}

// 학습 후 저장/로드 검증 (gradient 포함)
void test_trained_model_save_load_verify() {
    std::cout << "=== Test: Trained Model Save/Load Verification ===" << std::endl;

    // 1) 모델 생성 및 간단한 학습
    MLP model_original({4, 3});
    SGD optimizer;
    optimizer.setup(model_original);

    Tensor<> x_data({5, 4}, 1.0f);
    Tensor<> t_data({5}, std::vector<float>{0, 1, 2, 0, 1});
    Variable x(x_data);
    Variable t(t_data);

    // 몇 step 학습
    for (int i = 0; i < 3; ++i) {
        Variable y = model_original(x);
        Variable loss = softmax_cross_entropy_error(y, t);
        model_original.cleargrads();
        loss.backward();
        optimizer.update();
    }

    // 학습 후 forward 결과
    Variable y_original = model_original(x);
    std::cout << "  Training completed (3 steps)" << std::endl;

    // 2) 저장
    std::string weight_path = "test_trained_verify.dcz";
    model_original.save_weights(weight_path);
    std::cout << "  Saved trained weights" << std::endl;

    // 3) 새 모델에 로드
    MLP model_loaded({4, 3});
    Variable dummy = model_loaded(x);  // shape 초기화

    std::filesystem::path full_path = get_cache_file_path("weights/" + weight_path);
    model_loaded.load_weights(full_path.string());
    std::cout << "  Loaded weights into new model" << std::endl;

    // 4) forward 결과 비교
    Variable y_loaded = model_loaded(x);
    bool forward_match = tensors_equal(y_original.data(), y_loaded.data());

    std::cout << (forward_match ? "  [PASS] Trained model forward output matches"
                                : "  [FAIL] Trained model forward output mismatch") << std::endl;

    // 5) 각 파라미터 비교
    auto original_params = model_original.flatten_params();
    auto loaded_params = model_loaded.flatten_params();

    bool all_params_match = true;
    for (const auto& [name, param] : original_params) {
        auto it = loaded_params.find(name);
        if (it != loaded_params.end()) {
            bool match = tensors_equal(param.data(), it->second.data());
            std::cout << (match ? "  [PASS] " : "  [FAIL] ") << name << std::endl;
            all_params_match = all_params_match && match;
        }
    }

    bool all_passed = forward_match && all_params_match;
    std::cout << (all_passed ? "=== Trained Model Test PASSED ===" : "=== Trained Model Test FAILED ===") << std::endl;
    std::cout << std::endl;
}

int main() {
    test_linear_save_load_verify();
    test_conv2d_save_load_verify();
    test_mlp_save_load_verify();
    test_trained_model_save_load_verify();

    std::cout << "All weight verification tests completed." << std::endl;
    return 0;
}
