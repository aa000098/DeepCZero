#pragma once

#include "container/tensor/tensor_all.hpp"
#include "container/loss/yolov5_loss.hpp"
#include "utils/preprocess.hpp"

#include <string>
#include <vector>
#include <map>
#include <utility>

struct COCOImage {
	int id;
	std::string file_name;
	int width;
	int height;
};

class COCODataset {
private:
	std::string image_dir;
	size_t target_size;

	std::vector<COCOImage> images;
	std::map<int, GroundTruth> annotations;  // image_id -> GT boxes
	std::map<int, int> cat_id_to_idx;        // COCO category_id -> 0-based index

	void parse_annotations(const std::string& json_path);

public:
	COCODataset(const std::string& json_path,
				const std::string& image_dir,
				size_t target_size = 640);

	size_t size() const { return images.size(); }

	// Returns {image [1,3,H,W], ground_truth}
	std::pair<tensor::Tensor<>, GroundTruth> get_item(size_t index) const;
};
