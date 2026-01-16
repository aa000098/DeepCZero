#include "deepczero.hpp"
#include "utils/image.hpp"

#include <string>
#include <vector>
#include <algorithm>

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

	// 3. VGG16 forward
	VGG16 model(true);
	Variable y = model.forward({x});

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
