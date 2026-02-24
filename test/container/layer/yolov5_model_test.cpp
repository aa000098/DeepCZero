#include "deepczero.hpp"
#include <iostream>
#include <cassert>

void test_helpers() {
	std::cout << "=== Test: YOLOv5 width/depth helpers ===" << std::endl;

	// YOLOv5s: depth=0.33, width=0.50
	YOLOv5 model(80, 0.33f, 0.50f);

	// Verify model was created successfully
	std::cout << "  Model created (YOLOv5s, 80 classes)" << std::endl;
	std::cout << "  PASSED" << std::endl;
}

void test_model_construction() {
	std::cout << "=== Test: YOLOv5 model construction ===" << std::endl;

	YOLOv5 model(80, 0.33f, 0.50f);

	// Check that parameters exist
	auto params = model.get_params();
	std::cout << "  Total parameters: " << params.size() << std::endl;
	assert(params.size() > 0);

	std::cout << "  PASSED" << std::endl;
}

void test_forward_small() {
	std::cout << "=== Test: YOLOv5 forward pass (small input) ===" << std::endl;

	// Use small input to save memory: 64x64 instead of 640x640
	YOLOv5 model(80, 0.33f, 0.50f);

	// Create input: [1, 3, 64, 64]
	Tensor<> x_data({1, 3, 64, 64});
	auto& raw = x_data.raw_data();
	for (size_t i = 0; i < raw.size(); ++i)
		raw[i] = static_cast<float>(rand() % 256) / 255.0f;

	Variable x(x_data);

	dcz::UsingConfig eval_mode("train", false);
	auto outputs = model.forward_detect(x);

	assert(outputs.size() == 3);
	std::cout << "  Output count: " << outputs.size() << " (P3, P4, P5)" << std::endl;

	// With 64x64 input:
	// P3: [1, 255, 8, 8]   (64/8=8)
	// P4: [1, 255, 4, 4]   (64/16=4)
	// P5: [1, 255, 2, 2]   (64/32=2)
	size_t out_ch = 3 * (5 + 80);  // 255

	auto p3_shape = outputs[0].shape();
	auto p4_shape = outputs[1].shape();
	auto p5_shape = outputs[2].shape();

	std::cout << "  P3 shape: [";
	for (size_t i = 0; i < p3_shape.size(); ++i)
		std::cout << p3_shape[i] << (i < p3_shape.size()-1 ? ", " : "");
	std::cout << "]" << std::endl;

	std::cout << "  P4 shape: [";
	for (size_t i = 0; i < p4_shape.size(); ++i)
		std::cout << p4_shape[i] << (i < p4_shape.size()-1 ? ", " : "");
	std::cout << "]" << std::endl;

	std::cout << "  P5 shape: [";
	for (size_t i = 0; i < p5_shape.size(); ++i)
		std::cout << p5_shape[i] << (i < p5_shape.size()-1 ? ", " : "");
	std::cout << "]" << std::endl;

	// Verify shapes
	assert(p3_shape[0] == 1 && p3_shape[1] == out_ch && p3_shape[2] == 8 && p3_shape[3] == 8);
	assert(p4_shape[0] == 1 && p4_shape[1] == out_ch && p4_shape[2] == 4 && p4_shape[3] == 4);
	assert(p5_shape[0] == 1 && p5_shape[1] == out_ch && p5_shape[2] == 2 && p5_shape[3] == 2);

	std::cout << "  PASSED" << std::endl;
}

void test_forward_interface() {
	std::cout << "=== Test: YOLOv5 Layer interface (forward) ===" << std::endl;

	YOLOv5 model(80, 0.33f, 0.50f);

	Tensor<> x_data({1, 3, 64, 64});
	auto& raw = x_data.raw_data();
	for (size_t i = 0; i < raw.size(); ++i)
		raw[i] = 0.5f;

	Variable x(x_data);

	dcz::UsingConfig eval_mode("train", false);

	// Test standard Layer::operator()(Variable) interface
	Variable y = model(x);
	std::cout << "  forward() output shape: [";
	for (size_t i = 0; i < y.shape().size(); ++i)
		std::cout << y.shape()[i] << (i < y.shape().size()-1 ? ", " : "");
	std::cout << "]" << std::endl;

	// After forward(), detection_outputs should be available
	auto& det = model.get_detection_outputs();
	assert(det.size() == 3);
	std::cout << "  detection_outputs available: " << det.size() << " scales" << std::endl;

	std::cout << "  PASSED" << std::endl;
}

void test_custom_classes() {
	std::cout << "=== Test: YOLOv5 with custom num_classes ===" << std::endl;

	size_t num_classes = 20;  // VOC dataset
	YOLOv5 model(num_classes, 0.33f, 0.50f);

	Tensor<> x_data({1, 3, 32, 32});
	auto& raw = x_data.raw_data();
	for (size_t i = 0; i < raw.size(); ++i)
		raw[i] = 0.5f;

	Variable x(x_data);

	dcz::UsingConfig eval_mode("train", false);
	auto outputs = model.forward_detect(x);

	size_t expected_ch = 3 * (5 + num_classes);  // 75
	assert(outputs[0].shape()[1] == expected_ch);
	std::cout << "  Output channels for 20 classes: " << outputs[0].shape()[1]
			  << " (expected " << expected_ch << ")" << std::endl;

	std::cout << "  PASSED" << std::endl;
}

int main() {
	srand(42);

	std::cout << "=== YOLOv5 Model Tests ===" << std::endl;

	test_helpers();
	test_model_construction();
	test_forward_small();
	test_forward_interface();
	test_custom_classes();

	std::cout << "\nAll YOLOv5 model tests passed!" << std::endl;
	return 0;
}
