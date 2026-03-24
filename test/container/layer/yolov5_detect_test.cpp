#include "deepczero.hpp"
#include "utils/preprocess.hpp"
#include "utils/postprocess.hpp"

#include <string>
#include <iostream>
#include <chrono>

void test_yolov5_detection(const std::string& variant,
						   float depth_multiple, float width_multiple) {
	// Eval mode + no backprop
	dcz::UsingConfig eval_mode("train", false);
	dcz::UsingConfig no_grad("enable_backprop", false);

	// 1. Preprocess image (letterbox)
	std::string img_path = "sample/dog_bike_car.jpg";
	std::cout << "Image: " << img_path << std::endl;

	LetterboxInfo info;
	Tensor<> img_tensor = preprocess_yolov5(img_path, 640, &info);
	Variable x(img_tensor);

	std::cout << "Input shape: [";
	for (size_t i = 0; i < x.shape().size(); ++i)
		std::cout << x.shape()[i] << (i < x.shape().size()-1 ? ", " : "");
	std::cout << "]" << std::endl;
	std::cout << "Letterbox: scale=" << info.scale
			  << " pad=(" << info.pad_left << "," << info.pad_top << ")"
			  << " orig=" << info.orig_width << "x" << info.orig_height << std::endl;

	// 2. Load model
	std::cout << "\nLoading YOLOv5" << variant << " model..."
			  << " (depth=" << depth_multiple
			  << ", width=" << width_multiple << ")" << std::endl;
	YOLOv5 model(80, depth_multiple, width_multiple);

	std::string weights_path = std::string(getenv("HOME") ? getenv("HOME") : ".") +
							   "/.deepczero/weights/yolov5" + variant + ".npz";
	std::ifstream weight_check(weights_path);
	if (weight_check.good()) {
		weight_check.close();
		std::cout << "Loading weights from: " << weights_path << std::endl;
		model.load_weights(weights_path);
	} else {
		std::cout << "WARNING: Weights not found at " << weights_path << std::endl;
		std::cout << "Run: python scripts/convert_yolov5_weights.py yolov5"
				  << variant << ".pt" << std::endl;
		return;
	}

	// 3. Move model/input to GPU if available
#ifdef USE_SYCL
	dcz::SYCLContext::get().print_device_info();
	std::cout << "Moving model to GPU..." << std::endl;
	model.to(dcz::sycl());
	x = x.to(dcz::sycl());
#endif

	// 4. Forward pass
	std::cout << "\nRunning forward pass..." << std::endl;
	auto start = std::chrono::high_resolution_clock::now();
	auto outputs = model.forward_detect(x);
	auto end = std::chrono::high_resolution_clock::now();
	double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

	std::cout << "Forward pass: " << ms << " ms" << std::endl;

	// Move outputs to CPU for decode
	std::vector<Variable> cpu_outputs;
	for (auto& out : outputs) cpu_outputs.push_back(out.cpu());

	for (size_t i = 0; i < cpu_outputs.size(); ++i) {
		std::cout << "  Output " << i << " shape: [";
		for (size_t j = 0; j < cpu_outputs[i].shape().size(); ++j)
			std::cout << cpu_outputs[i].shape()[j] << (j < cpu_outputs[i].shape().size()-1 ? ", " : "");
		std::cout << "]" << std::endl;
	}

	// 5. Decode + NMS
	std::cout << "\nPost-processing..." << std::endl;
	auto detections = decode_yolov5_outputs(cpu_outputs, 80, 0.60f);
	std::cout << "Detections before NMS: " << detections.size() << std::endl;

	detections = nms(detections, 0.45f);
	std::cout << "Detections after NMS: " << detections.size() << std::endl;

	// 5. Rescale to original image coordinates
	rescale_detections(detections, info);

	// 6. Print results
	std::cout << "\n=== Detection Results ===" << std::endl;
	for (size_t i = 0; i < detections.size(); ++i) {
		const auto& det = detections[i];
		std::string cls_name = (det.class_id >= 0 &&
			det.class_id < static_cast<int>(COCO_CLASSES.size()))
			? COCO_CLASSES[det.class_id] : "unknown";

		std::cout << "  [" << i << "] " << cls_name
				  << " (conf: " << det.confidence << ")"
				  << " bbox: [" << det.x1 << ", " << det.y1
				  << ", " << det.x2 << ", " << det.y2 << "]"
				  << std::endl;
	}

	// 7. Draw and save output image
	if (!detections.empty()) {
		std::string output_path = "sample/yolov5_output.jpg";
		draw_detections(img_path, detections, output_path);
		std::cout << "\nOutput image saved to: " << output_path << std::endl;
	}
}

int main() {
	std::cout << "=== YOLOv5 End-to-End Detection Test ===" << std::endl;

	// YOLOv5s
	// test_yolov5_detection("s", 0.33f, 0.50f);

	// YOLOv5m
	test_yolov5_detection("m", 0.67f, 0.75f);

	std::cout << "\nDone!" << std::endl;
	return 0;
}
