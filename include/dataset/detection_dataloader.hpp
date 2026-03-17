#pragma once

#include "dataset/coco_dataset.hpp"

#include <vector>
#include <utility>

class DetectionDataLoader {
private:
	COCODataset& dataset;
	size_t batch_size;
	bool shuffle;
	size_t current_index;
	std::vector<size_t> indices;

public:
	DetectionDataLoader(COCODataset& dataset,
						size_t batch_size = 1,
						bool shuffle = true);

	void reset();
	bool has_next() const;

	// Returns {batched_images [B,3,H,W], vector<GroundTruth> of size B}
	std::pair<tensor::Tensor<>, std::vector<GroundTruth>> next_batch();

	size_t size() const { return indices.size(); }
};
