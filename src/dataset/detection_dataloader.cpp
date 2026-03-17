#include "dataset/detection_dataloader.hpp"

#include <algorithm>
#include <numeric>
#include <random>

using tensor::Tensor;


DetectionDataLoader::DetectionDataLoader(COCODataset& dataset,
										 size_t batch_size,
										 bool shuffle)
	: dataset(dataset), batch_size(batch_size), shuffle(shuffle), current_index(0) {
	indices.resize(dataset.size());
	std::iota(indices.begin(), indices.end(), 0);
	if (shuffle) {
		std::random_device rd;
		std::mt19937 g(rd());
		std::shuffle(indices.begin(), indices.end(), g);
	}
}


void DetectionDataLoader::reset() {
	current_index = 0;
	if (shuffle) {
		std::random_device rd;
		std::mt19937 g(rd());
		std::shuffle(indices.begin(), indices.end(), g);
	}
}


bool DetectionDataLoader::has_next() const {
	return current_index < indices.size();
}


std::pair<Tensor<>, std::vector<GroundTruth>> DetectionDataLoader::next_batch() {
	std::vector<Tensor<>> batch_images;
	std::vector<GroundTruth> batch_gt;

	for (size_t i = 0; i < batch_size && has_next(); i++, current_index++) {
		auto [image, gt] = dataset.get_item(indices[current_index]);
		batch_images.push_back(image);  // each is [1,3,H,W]
		batch_gt.push_back(gt);
	}

	// For batch_size=1, use image directly [1,3,H,W]
	// For batch_size>1, stack and reshape: remove leading 1 then stack
	Tensor<> images;
	if (batch_images.size() == 1) {
		images = batch_images[0];  // [1,3,H,W]
	} else {
		// Remove batch dim from each [1,C,H,W] -> [C,H,W], then stack -> [B,C,H,W]
		std::vector<Tensor<>> unbatched;
		auto shape = batch_images[0].get_shape();
		std::vector<size_t> squeeze_shape(shape.begin() + 1, shape.end());
		for (auto& img : batch_images) {
			unbatched.push_back(img.reshape(squeeze_shape));
		}
		images = tensor::stack(unbatched);
	}

	return {images, batch_gt};
}
