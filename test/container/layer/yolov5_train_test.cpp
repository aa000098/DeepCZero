#include "deepczero.hpp"
#include "utils/postprocess.hpp"
#include "utils/eval_metrics.hpp"
#include "dataset/coco_dataset.hpp"
#include "dataset/detection_dataloader.hpp"
#include "container/loss/yolov5_loss.hpp"

#include <iostream>
#include <chrono>
#include <string>


// Evaluate mAP on a dataset (batch_size=1)
float evaluate_map(YOLOv5& model, COCODataset& dataset,
				   size_t num_classes, size_t target_size, size_t max_images = 0) {
	dcz::UsingConfig eval_mode("train", false);
	dcz::UsingConfig no_grad("enable_backprop", false);

	DetectionDataLoader loader(dataset, 1, false);

	std::vector<std::vector<Detection>> all_preds;
	std::vector<std::vector<Detection>> all_gts;

	size_t count = 0;
	float ts = static_cast<float>(target_size);

	while (loader.has_next()) {
		if (max_images > 0 && count >= max_images) break;

		auto [images, gt_batch] = loader.next_batch();
		Variable x(images);
#ifdef USE_SYCL
		x = x.to(dcz::sycl());
#endif

		auto outputs = model.forward_detect(x);

		// Move outputs to CPU for decode
		std::vector<Variable> cpu_outputs;
		for (auto& out : outputs) cpu_outputs.push_back(out.cpu());

		// Decode predictions (in letterbox pixel coords)
		auto detections = decode_yolov5_outputs(cpu_outputs, num_classes, 0.001f);
		detections = nms(detections, 0.65f);
		all_preds.push_back(detections);

		// Convert GT from normalized [cls,cx,cy,w,h] to pixel [x1,y1,x2,y2]
		std::vector<Detection> gt_dets;
		for (const auto& box : gt_batch[0]) {
			Detection d;
			d.class_id = static_cast<int>(box[0]);
			float cx = box[1] * ts;
			float cy = box[2] * ts;
			float w = box[3] * ts;
			float h = box[4] * ts;
			d.x1 = cx - w / 2.0f;
			d.y1 = cy - h / 2.0f;
			d.x2 = cx + w / 2.0f;
			d.y2 = cy + h / 2.0f;
			d.confidence = 1.0f;
			gt_dets.push_back(d);
		}
		all_gts.push_back(gt_dets);

		count++;
		if (count % 100 == 0) {
			std::cout << "  Evaluated " << count << " images..." << std::endl;
		}
	}

	auto result = compute_map(all_preds, all_gts, num_classes, 0.5f);
	return result.map50;
}


void test_yolov5_training() {
	std::cout << "=== YOLOv5 Training Test ===" << std::endl;

	// Configuration
	size_t num_classes = 80;
	size_t target_size = 640;
	float lr = 0.001f;
	size_t num_epochs = 5;
	size_t log_interval = 1;

	// Paths (user should adjust these)
	std::string home = std::string(getenv("HOME") ? getenv("HOME") : ".");
	std::string weights_path = home + "/.deepczero/weights/yolov5s.npz";
	std::string train_json = "data/coco/annotations/instances_train2017.json";
	std::string train_images = "data/coco/images/train2017";
	std::string val_json = "data/coco/annotations/instances_val2017.json";
	std::string val_images = "data/coco/images/val2017";

	// 1. Load model
	std::cout << "\n[1] Loading YOLOv5s model..." << std::endl;
	YOLOv5 model(num_classes, 0.33f, 0.50f);

	std::ifstream weight_check(weights_path);
	if (weight_check.good()) {
		weight_check.close();
		model.load_weights(weights_path);
		std::cout << "  Loaded pretrained weights from: " << weights_path << std::endl;
	} else {
		std::cout << "  WARNING: No pretrained weights found. Training from scratch." << std::endl;
		std::cout << "  Run: python scripts/convert_yolov5_weights.py yolov5s.pt" << std::endl;
	}

#ifdef USE_SYCL
	dcz::SYCLContext::get().print_device_info();
	std::cout << "  Moving model to GPU..." << std::endl;
	model.to(dcz::sycl());
#endif

	// 2. Setup optimizer
	std::cout << "\n[2] Setting up optimizer (SGD lr=" << lr << ")..." << std::endl;
	MomentumSGD optimizer(lr, 0.937f);
	optimizer.setup(model);

	// 3. Setup loss
	YOLOv5Loss loss_fn(num_classes);

	// 4. Load dataset
	std::cout << "\n[3] Loading dataset..." << std::endl;
	COCODataset train_set(train_json, train_images, target_size);
	std::cout << "  Train set: " << train_set.size() << " images" << std::endl;

	DetectionDataLoader train_loader(train_set, 1, true);

	// 5. Training loop
	std::cout << "\n[4] Starting training..." << std::endl;
	for (size_t epoch = 0; epoch < num_epochs; epoch++) {
		std::cout << "\n--- Epoch " << (epoch + 1) << "/" << num_epochs << " ---" << std::endl;
		dcz::Config::get().train = true;

		train_loader.reset();
		float epoch_box = 0, epoch_obj = 0, epoch_cls = 0;
		size_t step = 0;

		auto epoch_start = std::chrono::high_resolution_clock::now();

		while (train_loader.has_next()) {
			auto [images, gt] = train_loader.next_batch();
			Variable x(images);
#ifdef USE_SYCL
			x = x.to(dcz::sycl());
#endif

			// Forward
			auto preds = model.forward_detect(x);

			// Move to CPU for loss computation
			std::vector<Variable> cpu_preds;
			for (auto& p : preds) cpu_preds.push_back(p.cpu());

			// Loss
			auto result = loss_fn(cpu_preds, gt);

			// Backward
			model.cleargrads();
			result.total_loss.backward();

			// Update
			optimizer.update();

			epoch_box += result.box_loss_val;
			epoch_obj += result.obj_loss_val;
			epoch_cls += result.cls_loss_val;
			step++;

			if (step % log_interval == 0) {
				std::cout << "  Step " << step
						  << "  box=" << (epoch_box / step)
						  << "  obj=" << (epoch_obj / step)
						  << "  cls=" << (epoch_cls / step)
						  << std::endl;
			}
		}

		auto epoch_end = std::chrono::high_resolution_clock::now();
		double epoch_sec = std::chrono::duration_cast<std::chrono::seconds>(
			epoch_end - epoch_start).count();

		std::cout << "Epoch " << (epoch + 1) << " done (" << epoch_sec << "s)"
				  << "  avg_box=" << (epoch_box / step)
				  << "  avg_obj=" << (epoch_obj / step)
				  << "  avg_cls=" << (epoch_cls / step)
				  << std::endl;
	}

	std::cout << "\nTraining complete!" << std::endl;
}


// Smoke test: overfit on a tiny dataset to verify loss decreases
void test_overfit_small() {
	std::cout << "=== YOLOv5 Overfit Smoke Test ===" << std::endl;

	size_t num_classes = 80;
	size_t target_size = 640;

	std::string home = std::string(getenv("HOME") ? getenv("HOME") : ".");
	std::string weights_path = home + "/.deepczero/weights/yolov5s.npz";
	std::string json_path = "data/coco/annotations/instances_val2017.json";
	std::string image_dir = "data/coco/images/val2017";

	// Load model
	YOLOv5 model(num_classes, 0.33f, 0.50f);
	std::ifstream weight_check(weights_path);
	if (weight_check.good()) {
		weight_check.close();
		model.load_weights(weights_path);
		std::cout << "Loaded pretrained weights" << std::endl;
	} else {
		std::cout << "No weights found, skipping test." << std::endl;
		return;
	}

#ifdef USE_SYCL
	dcz::SYCLContext::get().print_device_info();
	std::cout << "Moving model to GPU..." << std::endl;
	model.to(dcz::sycl());
#endif

	MomentumSGD optimizer(0.001f, 0.937f);
	optimizer.setup(model);

	YOLOv5Loss loss_fn(num_classes);
	COCODataset dataset(json_path, image_dir, target_size);
	DetectionDataLoader loader(dataset, 1, false);

	// Train for a few steps on first N images
	size_t max_steps = 10;
	std::cout << "\nOverfitting on " << max_steps << " images for 3 epochs..." << std::endl;

	for (size_t epoch = 0; epoch < 3; epoch++) {
		loader.reset();
		float total_loss = 0;
		size_t step = 0;

		dcz::Config::get().train = true;

		while (loader.has_next() && step < max_steps) {
			auto [images, gt] = loader.next_batch();
			Variable x(images);
#ifdef USE_SYCL
			x = x.to(dcz::sycl());
#endif

			auto preds = model.forward_detect(x);

			// Move to CPU for loss computation
			std::vector<Variable> cpu_preds;
			for (auto& p : preds) cpu_preds.push_back(p.cpu());

			auto result = loss_fn(cpu_preds, gt);

			model.cleargrads();
			result.total_loss.backward();
			optimizer.update();

			total_loss += result.box_loss_val + result.obj_loss_val + result.cls_loss_val;
			step++;
		}

		std::cout << "  Epoch " << (epoch + 1) << ": avg_loss = " << (total_loss / step) << std::endl;
	}

	std::cout << "\nOverfit test done. Loss should decrease across epochs." << std::endl;
}


int main() {
	// Smoke test (small, quick)
	test_overfit_small();

	// Full training (uncomment when ready)
	test_yolov5_training();

	return 0;
}
