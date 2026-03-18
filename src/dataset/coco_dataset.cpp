#include "dataset/coco_dataset.hpp"

#include "json.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using json = nlohmann::json;
using tensor::Tensor;


COCODataset::COCODataset(const std::string& json_path,
						 const std::string& image_dir,
						 size_t target_size)
	: image_dir(image_dir), target_size(target_size) {
	parse_annotations(json_path);
}


void COCODataset::parse_annotations(const std::string& json_path) {
	std::ifstream f(json_path);
	if (!f.is_open()) {
		throw std::runtime_error("Cannot open COCO annotation file: " + json_path);
	}

	json data = json::parse(f);

	// Build category_id -> 0-based index mapping
	const auto& categories = data["categories"];
	for (size_t i = 0; i < categories.size(); i++) {
		int cat_id = categories[i]["id"].get<int>();
		cat_id_to_idx[cat_id] = static_cast<int>(i);
	}

	// Build image_id -> COCOImage mapping + collect image list
	std::map<int, size_t> id_to_img_idx;
	const auto& imgs = data["images"];
	images.reserve(imgs.size());
	for (const auto& img : imgs) {
		COCOImage ci;
		ci.id = img["id"].get<int>();
		ci.file_name = img["file_name"].get<std::string>();
		ci.width = img["width"].get<int>();
		ci.height = img["height"].get<int>();
		id_to_img_idx[ci.id] = images.size();
		images.push_back(ci);
	}

	// Parse annotations: convert bbox [x,y,w,h] pixels -> normalized [cls,cx,cy,w,h]
	const auto& anns = data["annotations"];
	for (const auto& ann : anns) {
		int image_id = ann["image_id"].get<int>();
		int cat_id = ann["category_id"].get<int>();

		if (cat_id_to_idx.find(cat_id) == cat_id_to_idx.end()) continue;
		if (id_to_img_idx.find(image_id) == id_to_img_idx.end()) continue;

		// Skip crowd annotations
		if (ann.contains("iscrowd") && ann["iscrowd"].get<int>() == 1) continue;

		const auto& bbox = ann["bbox"];
		float x = bbox[0].get<float>();
		float y = bbox[1].get<float>();
		float w = bbox[2].get<float>();
		float h = bbox[3].get<float>();

		// Skip degenerate boxes
		if (w <= 0 || h <= 0) continue;

		size_t img_idx = id_to_img_idx[image_id];
		float img_w = static_cast<float>(images[img_idx].width);
		float img_h = static_cast<float>(images[img_idx].height);

		float cls = static_cast<float>(cat_id_to_idx[cat_id]);
		float cx = (x + w / 2.0f) / img_w;
		float cy = (y + h / 2.0f) / img_h;
		float nw = w / img_w;
		float nh = h / img_h;

		annotations[image_id].push_back({cls, cx, cy, nw, nh});
	}

	std::cout << "[COCODataset] Loaded " << images.size() << " images, "
			  << annotations.size() << " annotated images" << std::endl;
}


std::pair<Tensor<>, GroundTruth> COCODataset::get_item(size_t index) const {
	const COCOImage& img = images[index];
	std::string path = image_dir + "/" + img.file_name;

	// Load and preprocess image with letterbox
	LetterboxInfo info;
	Tensor<> image = preprocess_yolov5(path, target_size, &info);

	// Adjust GT boxes from original normalized coords to letterbox coords
	GroundTruth gt;
	auto it = annotations.find(img.id);
	if (it != annotations.end()) {
		float ts = static_cast<float>(target_size);
		for (const auto& box : it->second) {
			float cls = box[0];
			float cx = box[1];  // normalized to original image
			float cy = box[2];
			float w = box[3];
			float h = box[4];

			// Convert to letterbox coordinates
			float new_cx = (cx * static_cast<float>(info.orig_width) * info.scale
							+ static_cast<float>(info.pad_left)) / ts;
			float new_cy = (cy * static_cast<float>(info.orig_height) * info.scale
							+ static_cast<float>(info.pad_top)) / ts;
			float new_w = w * static_cast<float>(info.orig_width) * info.scale / ts;
			float new_h = h * static_cast<float>(info.orig_height) * info.scale / ts;

			// Clip to [0, 1]
			new_cx = std::clamp(new_cx, 0.0f, 1.0f);
			new_cy = std::clamp(new_cy, 0.0f, 1.0f);
			new_w = std::min(new_w, 1.0f);
			new_h = std::min(new_h, 1.0f);

			if (new_w > 0.001f && new_h > 0.001f) {
				gt.push_back({cls, new_cx, new_cy, new_w, new_h});
			}
		}
	}

	return {image, gt};
}
