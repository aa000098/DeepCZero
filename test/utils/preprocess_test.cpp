#include "deepczero.hpp"
#include "utils/image.hpp"

void test_preprocess_only() {
	std::cout << "=== Test: VGG16 Preprocessing (C++) ===" << std::endl;

	// 이미지 다운로드
	std::string file_name = "zebra.jpg";
	std::string zebra_image_url = "https://github.com/Wegralee/deep-learning-from-scratch-3/raw/images/" + file_name;
	std::string zebra_file_path = get_file(zebra_image_url, "data/" + file_name);

	std::cout << "\nImage path: " << zebra_file_path << std::endl;

	// 이미지 전처리
	Tensor<> img_tensor = preprocess_vgg16(zebra_file_path);

	std::cout << "\n=== Preprocessed Result ===" << std::endl;
	std::cout << "Shape: ";
	for (auto s : img_tensor.get_shape()) std::cout << s << " ";
	std::cout << std::endl;

	const auto& data = img_tensor.raw_data();
	float min_val = *std::min_element(data.begin(), data.end());
	float max_val = *std::max_element(data.begin(), data.end());
	float mean_val = std::accumulate(data.begin(), data.end(), 0.0f) / data.size();

	std::cout << "Min: " << min_val << ", Max: " << max_val << ", Mean: " << mean_val << std::endl;

	std::cout << "\n=== Sample pixels [0, :, 0, 0] ===" << std::endl;
	std::cout << "B channel: " << img_tensor({0, 0, 0, 0}) << std::endl;
	std::cout << "G channel: " << img_tensor({0, 1, 0, 0}) << std::endl;
	std::cout << "R channel: " << img_tensor({0, 2, 0, 0}) << std::endl;

	std::cout << "\n=== First 5 pixels of B channel [0, 0, 0, :5] ===" << std::endl;
	for (int i = 0; i < 5; ++i) {
		std::cout << img_tensor({0, 0, 0, static_cast<size_t>(i)}) << " ";
	}
	std::cout << std::endl;

	std::cout << "\n=== Center pixel [0, :, 112, 112] ===" << std::endl;
	std::cout << "B: " << img_tensor({0, 0, 112, 112}) << std::endl;
	std::cout << "G: " << img_tensor({0, 1, 112, 112}) << std::endl;
	std::cout << "R: " << img_tensor({0, 2, 112, 112}) << std::endl;

	std::cout << "\n=== Test completed ===" << std::endl;
}

int main() {
	test_preprocess_only();
	return 0;
}
