#include "deepczero.hpp"
#include "utils/image.hpp"

#include <string>
#include <vector>
#include <algorithm>
#include <numeric>

void test_vgg_forward_with_image() {
	dcz::UsingConfig test_mode("train", false);

	// 0. ImageNet 레이블 로드
	ImageNetDataset imagenet_labels;

	// 1. 이미지 다운로드
	std::string file_name = "zebra.jpg";
	std::string zebra_image_url = "https://github.com/Wegralee/deep-learning-from-scratch-3/raw/images/" + file_name;
	std::string zebra_file_path = get_file(zebra_image_url, "data/" + file_name);

	std::cout << "Image path: " << zebra_file_path << std::endl;

	// 2. 이미지 전처리
	Tensor<> img_tensor = preprocess_vgg16(zebra_file_path);
	Variable x(img_tensor);

	std::cout << "\n=== Input Statistics ===" << std::endl;
	const auto& input_data = x.data().raw_data();
	float input_min = *std::min_element(input_data.begin(), input_data.end());
	float input_max = *std::max_element(input_data.begin(), input_data.end());
	float input_mean = std::accumulate(input_data.begin(), input_data.end(), 0.0f) / input_data.size();
	std::cout << "  Min: " << input_min << ", Max: " << input_max << ", Mean: " << input_mean << std::endl;

	std::cout << "\n=== Sample pixels [0, :, 0, 0] ===" << std::endl;
	std::cout << "  B channel: " << x.data()({0, 0, 0, 0}) << std::endl;
	std::cout << "  G channel: " << x.data()({0, 1, 0, 0}) << std::endl;
	std::cout << "  R channel: " << x.data()({0, 2, 0, 0}) << std::endl;

	std::cout << "\n=== First 5 pixels of B channel [0, 0, 0, :5] ===" << std::endl;
	std::cout << "  ";
	for (int i = 0; i < 5; ++i) {
		std::cout << x.data()({0, 0, 0, static_cast<size_t>(i)}) << " ";
	}
	std::cout << std::endl;

	std::cout << "\n=== Center pixel [0, :, 112, 112] ===" << std::endl;
	std::cout << "  B: " << x.data()({0, 0, 112, 112}) << std::endl;
	std::cout << "  G: " << x.data()({0, 1, 112, 112}) << std::endl;
	std::cout << "  R: " << x.data()({0, 2, 112, 112}) << std::endl;

	// 3. VGG16 forward
	VGG16 model(true);
	Variable y = model.forward({x});

	std::cout << "\n=== Output Statistics ===" << std::endl;
	const auto& output_data = y.data().raw_data();
	float output_min = *std::min_element(output_data.begin(), output_data.end());
	float output_max = *std::max_element(output_data.begin(), output_data.end());
	float output_mean = std::accumulate(output_data.begin(), output_data.end(), 0.0f) / output_data.size();
	std::cout << "  Min: " << output_min << ", Max: " << output_max << ", Mean: " << output_mean << std::endl;

	// 4. 결과 출력
	std::cout << "\n=== VGG16 Prediction ===" << std::endl;
	std::cout << "Output shape: ";
	for (auto s : y.shape()) std::cout << s << " ";
	std::cout << "\nTop-5 predictions:" << std::endl;

	// Top-5 계산
	const auto& logits = y.data().raw_data();
	std::vector<std::pair<int, float>> scores;
	for (size_t i = 0; i < logits.size(); ++i) {
		scores.push_back({static_cast<int>(i), logits[i]});
	}

	// 내림차순 정렬
	std::sort(scores.begin(), scores.end(),
		[](const auto& a, const auto& b) { return a.second > b.second; });

	// Top-5 출력
	for (int i = 0; i < 5 && i < static_cast<int>(scores.size()); ++i) {
		int class_idx = scores[i].first;
		float score = scores[i].second;
		std::string label = imagenet_labels.labels(class_idx);
		std::cout << "  " << (i+1) << ". Class " << class_idx
				  << " (" << label << "): " << score << std::endl;
	}
}
                
 
int main() {
	test_vgg_forward_with_image();
}
