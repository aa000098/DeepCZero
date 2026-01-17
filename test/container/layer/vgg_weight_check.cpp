#include "deepczero.hpp"
#include <numeric>

void test_vgg_weight_check() {
	std::cout << "=== Test: VGG16 Weight Verification ===" << std::endl;

	// pretrained weights 로드
	std::cout << "\n1. Creating VGG16 with pretrained weights..." << std::endl;
	VGG16 model(true);
	std::cout << "   Model created" << std::endl;

	// conv1_1 가중치 확인
	std::cout << "\n2. Checking conv1_1 weights..." << std::endl;
	auto conv1_1 = model.get_layer("conv1_1");
	if (!conv1_1) {
		std::cout << "   ERROR: conv1_1 not found!" << std::endl;
		return;
	}

	Parameter W = conv1_1->get_param("W");
	Parameter b = conv1_1->get_param("b");

	std::cout << "   W shape: ";
	for (auto s : W.shape()) std::cout << s << " ";
	std::cout << std::endl;

	std::cout << "   b shape: ";
	for (auto s : b.shape()) std::cout << s << " ";
	std::cout << std::endl;

	const auto& w_data = W.data().raw_data();
	const auto& b_data = b.data().raw_data();

	std::cout << "   W first 5 values: ";
	for (int i = 0; i < 5 && i < static_cast<int>(w_data.size()); ++i) {
		std::cout << w_data[i] << " ";
	}
	std::cout << std::endl;

	float w_mean = std::accumulate(w_data.begin(), w_data.end(), 0.0f) / w_data.size();
	float b_mean = std::accumulate(b_data.begin(), b_data.end(), 0.0f) / b_data.size();
	std::cout << "   W mean: " << w_mean << ", b mean: " << b_mean << std::endl;

	// fc6 가중치 확인
	std::cout << "\n3. Checking fc6 weights..." << std::endl;
	auto fc6 = model.get_layer("fc6");
	if (!fc6) {
		std::cout << "   ERROR: fc6 not found!" << std::endl;
		return;
	}

	Parameter fc6_W = fc6->get_param("W");
	Parameter fc6_b = fc6->get_param("b");

	std::cout << "   W shape: ";
	for (auto s : fc6_W.shape()) std::cout << s << " ";
	std::cout << std::endl;

	std::cout << "   b shape: ";
	for (auto s : fc6_b.shape()) std::cout << s << " ";
	std::cout << std::endl;

	const auto& fc6_w_data = fc6_W.data().raw_data();
	const auto& fc6_b_data = fc6_b.data().raw_data();

	std::cout << "   W first 5 values: ";
	for (int i = 0; i < 5 && i < static_cast<int>(fc6_w_data.size()); ++i) {
		std::cout << fc6_w_data[i] << " ";
	}
	std::cout << std::endl;

	float fc6_w_mean = std::accumulate(fc6_w_data.begin(), fc6_w_data.end(), 0.0f) / fc6_w_data.size();
	float fc6_b_mean = std::accumulate(fc6_b_data.begin(), fc6_b_data.end(), 0.0f) / fc6_b_data.size();
	std::cout << "   W mean: " << fc6_w_mean << ", b mean: " << fc6_b_mean << std::endl;

	std::cout << "\n=== Testing completed ===" << std::endl;
}

int main() {
	test_vgg_weight_check();
	return 0;
}
