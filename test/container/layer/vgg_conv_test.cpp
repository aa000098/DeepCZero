#include "deepczero.hpp"
#include <numeric>

void test_conv1_1_forward() {
	std::cout << "=== Test: VGG16 Conv1_1 Forward ===" << std::endl;

	dcz::UsingConfig test_mode("train", false);

	// 1. 간단한 입력 생성 (모두 1로 채움)
	std::cout << "\n1. Creating simple input (all ones)..." << std::endl;
	Tensor<> x_data({1, 3, 224, 224}, 1.0f);
	Variable x(x_data);

	std::cout << "   Input shape: ";
	for (auto s : x.shape()) std::cout << s << " ";
	std::cout << std::endl;

	// 2. VGG16 모델 로드 및 conv1_1만 실행
	std::cout << "\n2. Loading VGG16 and running conv1_1..." << std::endl;
	VGG16 model(true);

	auto conv1_1 = model.get_layer("conv1_1");
	Parameter W = conv1_1->get_param("W");
	Parameter b = conv1_1->get_param("b");

	std::cout << "   W shape: ";
	for (auto s : W.shape()) std::cout << s << " ";
	std::cout << std::endl;

	std::cout << "   b shape: ";
	for (auto s : b.shape()) std::cout << s << " ";
	std::cout << std::endl;

	// conv1_1 forward
	Variable y = (*conv1_1)({x});

	std::cout << "\n3. Conv1_1 output..." << std::endl;
	std::cout << "   Output shape: ";
	for (auto s : y.shape()) std::cout << s << " ";
	std::cout << std::endl;

	const auto& output_data = y.data().raw_data();
	float output_min = *std::min_element(output_data.begin(), output_data.end());
	float output_max = *std::max_element(output_data.begin(), output_data.end());
	float output_mean = std::accumulate(output_data.begin(), output_data.end(), 0.0f) / output_data.size();

	std::cout << "   Output min: " << output_min << std::endl;
	std::cout << "   Output max: " << output_max << std::endl;
	std::cout << "   Output mean: " << output_mean << std::endl;

	// 첫 번째 출력 채널의 첫 픽셀
	std::cout << "   First pixel [0, 0, 0, 0]: " << y.data()({0, 0, 0, 0}) << std::endl;
	std::cout << "   Center pixel [0, 0, 112, 112]: " << y.data()({0, 0, 112, 112}) << std::endl;

	std::cout << "\n=== Test completed ===" << std::endl;
}

int main() {
	test_conv1_1_forward();
	return 0;
}
