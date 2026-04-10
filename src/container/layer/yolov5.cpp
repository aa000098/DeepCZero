#include "container/layer/yolov5.hpp"
#include "function/ops/ops_all.hpp"
#include "container/variable_ops.hpp"
#include "utils/io.hpp"
#include "cnpy.h"

namespace layer {

// ===== CBS (Conv-BatchNorm-SiLU) =====

CBS::CBS(size_t in_channels, size_t out_channels,
		 std::pair<size_t, size_t> kernel_size,
		 std::pair<size_t, size_t> stride,
		 std::pair<size_t, size_t> pad) {
	conv = std::make_shared<Conv2d>(out_channels, kernel_size, stride, pad,
									/*no_bias=*/true, in_channels);
	bn = std::make_shared<BatchNorm2d>(out_channels, 0.03f, 0.001f);

	register_sublayers("conv", conv);
	register_sublayers("bn", bn);
}

Variable CBS::forward(const std::vector<Variable>& xs) {
	Variable y = (*conv)(xs[0]);
	y = (*bn)(y);
	y = silu(y);
	return y;
}

void CBS::load_from_npz(const cnpy::npz_t& npz, const std::string& prefix) {
	conv->load_params_from_npz(npz, prefix + ".conv");
	bn->load_from_npz(npz, prefix + ".bn");
}

// ===== Bottleneck =====

Bottleneck::Bottleneck(size_t in_channels, size_t out_channels,
					   bool shortcut, float expansion)
	: shortcut(shortcut && in_channels == out_channels) {
	size_t hidden = static_cast<size_t>(out_channels * expansion);

	cv1 = std::make_shared<CBS>(in_channels, hidden, std::make_pair(1u, 1u),
								std::make_pair(1u, 1u), std::make_pair(0u, 0u));
	cv2 = std::make_shared<CBS>(hidden, out_channels, std::make_pair(3u, 3u),
								std::make_pair(1u, 1u), std::make_pair(1u, 1u));

	register_sublayers("cv1", cv1);
	register_sublayers("cv2", cv2);
}

Variable Bottleneck::forward(const std::vector<Variable>& xs) {
	Variable y = (*cv2)((*cv1)(xs[0]));
	if (shortcut)
		y = y + xs[0];
	return y;
}

void Bottleneck::load_from_npz(const cnpy::npz_t& npz, const std::string& prefix) {
	cv1->load_from_npz(npz, prefix + ".cv1");
	cv2->load_from_npz(npz, prefix + ".cv2");
}

// ===== C3 (CSP Bottleneck with 3 convolutions) =====

C3::C3(size_t in_channels, size_t out_channels,
	   size_t n, bool shortcut, float expansion)
	: num_bottlenecks(n) {
	size_t hidden = static_cast<size_t>(out_channels * expansion);

	cv1 = std::make_shared<CBS>(in_channels, hidden, std::make_pair(1u, 1u),
								std::make_pair(1u, 1u), std::make_pair(0u, 0u));
	cv2 = std::make_shared<CBS>(in_channels, hidden, std::make_pair(1u, 1u),
								std::make_pair(1u, 1u), std::make_pair(0u, 0u));
	cv3 = std::make_shared<CBS>(hidden * 2, out_channels, std::make_pair(1u, 1u),
								std::make_pair(1u, 1u), std::make_pair(0u, 0u));

	register_sublayers("cv1", cv1);
	register_sublayers("cv2", cv2);
	register_sublayers("cv3", cv3);

	for (size_t i = 0; i < n; ++i) {
		auto bn = std::make_shared<Bottleneck>(hidden, hidden, shortcut);
		bottlenecks.push_back(bn);
		register_sublayers("m_" + std::to_string(i), bn);
	}
}

Variable C3::forward(const std::vector<Variable>& xs) {
	Variable y1 = (*cv1)(xs[0]);
	// Pass through bottleneck chain
	for (size_t i = 0; i < num_bottlenecks; ++i)
		y1 = (*bottlenecks[i])(y1);

	Variable y2 = (*cv2)(xs[0]);

	// Concatenate along channel axis
	Variable y = concat({y1, y2}, 1);
	y = (*cv3)(y);
	return y;
}

void C3::load_from_npz(const cnpy::npz_t& npz, const std::string& prefix) {
	cv1->load_from_npz(npz, prefix + ".cv1");
	cv2->load_from_npz(npz, prefix + ".cv2");
	cv3->load_from_npz(npz, prefix + ".cv3");
	for (size_t i = 0; i < bottlenecks.size(); ++i) {
		bottlenecks[i]->load_from_npz(npz, prefix + ".m." + std::to_string(i));
	}
}

// ===== SPPF (Spatial Pyramid Pooling - Fast) =====

SPPF::SPPF(size_t in_channels, size_t out_channels, size_t k)
	: pool_size({k, k}) {
	size_t hidden = in_channels / 2;

	cv1 = std::make_shared<CBS>(in_channels, hidden, std::make_pair(1u, 1u),
								std::make_pair(1u, 1u), std::make_pair(0u, 0u));
	cv2 = std::make_shared<CBS>(hidden * 4, out_channels, std::make_pair(1u, 1u),
								std::make_pair(1u, 1u), std::make_pair(0u, 0u));

	register_sublayers("cv1", cv1);
	register_sublayers("cv2", cv2);
}

Variable SPPF::forward(const std::vector<Variable>& xs) {
	Variable x = (*cv1)(xs[0]);

	// MaxPool with stride=1 and padding to maintain spatial size
	size_t k = pool_size.first;
	size_t pad = k / 2;
	auto pool_k = pool_size;
	std::pair<size_t, size_t> pool_s = {1, 1};
	std::pair<size_t, size_t> pool_p = {pad, pad};

	Variable y1 = pooling(x, pool_k, pool_s, pool_p);
	Variable y2 = pooling(y1, pool_k, pool_s, pool_p);
	Variable y3 = pooling(y2, pool_k, pool_s, pool_p);

	Variable y = concat({x, y1, y2, y3}, 1);
	y = (*cv2)(y);
	return y;
}

void SPPF::load_from_npz(const cnpy::npz_t& npz, const std::string& prefix) {
	cv1->load_from_npz(npz, prefix + ".cv1");
	cv2->load_from_npz(npz, prefix + ".cv2");
}

} // namespace layer


// ===== YOLOv5 Model =====

size_t YOLOv5::make_divisible(size_t v, size_t divisor) {
	return ((v + divisor / 2) / divisor) * divisor;
}

size_t YOLOv5::width(size_t channels) const {
	return make_divisible(static_cast<size_t>(channels * width_multiple), 8);
}

size_t YOLOv5::depth(size_t n) const {
	return std::max(static_cast<size_t>(std::round(n * depth_multiple)),
					static_cast<size_t>(1));
}

YOLOv5::YOLOv5(size_t num_classes, float depth_multiple, float width_multiple, bool pretrained)
	: depth_multiple(depth_multiple), width_multiple(width_multiple) {

	size_t out_ch = NUM_ANCHORS * (5 + num_classes);

	// Backbone
	backbone_0 = std::make_shared<layer::CBS>(3, width(64),
					std::make_pair(6u, 6u), std::make_pair(2u, 2u), std::make_pair(2u, 2u));
	backbone_1 = std::make_shared<layer::CBS>(width(64), width(128),
					std::make_pair(3u, 3u), std::make_pair(2u, 2u), std::make_pair(1u, 1u));
	backbone_2 = std::make_shared<layer::C3>(width(128), width(128), depth(3));
	backbone_3 = std::make_shared<layer::CBS>(width(128), width(256),
					std::make_pair(3u, 3u), std::make_pair(2u, 2u), std::make_pair(1u, 1u));
	backbone_4 = std::make_shared<layer::C3>(width(256), width(256), depth(6));
	backbone_5 = std::make_shared<layer::CBS>(width(256), width(512),
					std::make_pair(3u, 3u), std::make_pair(2u, 2u), std::make_pair(1u, 1u));
	backbone_6 = std::make_shared<layer::C3>(width(512), width(512), depth(9));
	backbone_7 = std::make_shared<layer::CBS>(width(512), width(1024),
					std::make_pair(3u, 3u), std::make_pair(2u, 2u), std::make_pair(1u, 1u));
	backbone_8 = std::make_shared<layer::C3>(width(1024), width(1024), depth(3));
	backbone_9 = std::make_shared<layer::SPPF>(width(1024), width(1024), 5);

	// Neck - top-down
	neck_10 = std::make_shared<layer::CBS>(width(1024), width(512),
					std::make_pair(1u, 1u), std::make_pair(1u, 1u), std::make_pair(0u, 0u));
	neck_13 = std::make_shared<layer::C3>(width(1024), width(512), depth(3), false);
	neck_14 = std::make_shared<layer::CBS>(width(512), width(256),
					std::make_pair(1u, 1u), std::make_pair(1u, 1u), std::make_pair(0u, 0u));
	neck_17 = std::make_shared<layer::C3>(width(512), width(256), depth(3), false);

	// Neck - bottom-up
	neck_18 = std::make_shared<layer::CBS>(width(256), width(256),
					std::make_pair(3u, 3u), std::make_pair(2u, 2u), std::make_pair(1u, 1u));
	neck_20 = std::make_shared<layer::C3>(width(512), width(512), depth(3), false);
	neck_21 = std::make_shared<layer::CBS>(width(512), width(512),
					std::make_pair(3u, 3u), std::make_pair(2u, 2u), std::make_pair(1u, 1u));
	neck_23 = std::make_shared<layer::C3>(width(1024), width(1024), depth(3), false);

	// Detection heads (raw Conv2d, 1x1, with bias)
	detect_p3 = std::make_shared<layer::Conv2d>(out_ch,
					std::make_pair(1u, 1u), std::make_pair(1u, 1u), std::make_pair(0u, 0u),
					false, width(256));
	detect_p4 = std::make_shared<layer::Conv2d>(out_ch,
					std::make_pair(1u, 1u), std::make_pair(1u, 1u), std::make_pair(0u, 0u),
					false, width(512));
	detect_p5 = std::make_shared<layer::Conv2d>(out_ch,
					std::make_pair(1u, 1u), std::make_pair(1u, 1u), std::make_pair(0u, 0u),
					false, width(1024));

	// Register all sublayers
	register_sublayers("backbone_0", backbone_0);
	register_sublayers("backbone_1", backbone_1);
	register_sublayers("backbone_2", backbone_2);
	register_sublayers("backbone_3", backbone_3);
	register_sublayers("backbone_4", backbone_4);
	register_sublayers("backbone_5", backbone_5);
	register_sublayers("backbone_6", backbone_6);
	register_sublayers("backbone_7", backbone_7);
	register_sublayers("backbone_8", backbone_8);
	register_sublayers("backbone_9", backbone_9);
	register_sublayers("neck_10", neck_10);
	register_sublayers("neck_13", neck_13);
	register_sublayers("neck_14", neck_14);
	register_sublayers("neck_17", neck_17);
	register_sublayers("neck_18", neck_18);
	register_sublayers("neck_20", neck_20);
	register_sublayers("neck_21", neck_21);
	register_sublayers("neck_23", neck_23);
	register_sublayers("detect_p3", detect_p3);
	register_sublayers("detect_p4", detect_p4);
	register_sublayers("detect_p5", detect_p5);

	if (pretrained) {
		std::string weights_path = get_file("", "weights/yolov5s.npz");
		load_weights(weights_path);
	}
}

std::vector<Variable> YOLOv5::forward_detect(const Variable& x) {
	// Backbone
	Variable b0 = (*backbone_0)(x);    // /2
	Variable b1 = (*backbone_1)(b0);   // /4
	Variable b2 = (*backbone_2)(b1);   // /4
	Variable b3 = (*backbone_3)(b2);   // /8
	Variable b4 = (*backbone_4)(b3);   // /8  (save for P3 concat)
	Variable b5 = (*backbone_5)(b4);   // /16
	Variable b6 = (*backbone_6)(b5);   // /16 (save for P4 concat)
	Variable b7 = (*backbone_7)(b6);   // /32
	Variable b8 = (*backbone_8)(b7);   // /32
	Variable b9 = (*backbone_9)(b8);   // /32

	// Neck - top-down path
	Variable n10 = (*neck_10)(b9);                  // 1x1 conv
	Variable n11 = upsample(n10, 2);                // upsample x2
	Variable n12 = concat({n11, b6}, 1);            // concat with backbone_6
	Variable n13 = (*neck_13)(n12);                 // C3
	Variable n14 = (*neck_14)(n13);                 // 1x1 conv
	Variable n15 = upsample(n14, 2);                // upsample x2
	Variable n16 = concat({n15, b4}, 1);            // concat with backbone_4
	Variable n17 = (*neck_17)(n16);                 // C3 -> P3 feature

	// Neck - bottom-up path
	Variable n18 = (*neck_18)(n17);                 // CBS stride=2
	Variable n19 = concat({n18, n14}, 1);           // concat with neck_14
	Variable n20 = (*neck_20)(n19);                 // C3 -> P4 feature
	Variable n21 = (*neck_21)(n20);                 // CBS stride=2
	Variable n22 = concat({n21, n10}, 1);           // concat with neck_10
	Variable n23 = (*neck_23)(n22);                 // C3 -> P5 feature

	// Detection heads
	Variable p3 = (*detect_p3)(n17);
	Variable p4 = (*detect_p4)(n20);
	Variable p5 = (*detect_p5)(n23);

	detection_outputs = {p3, p4, p5};
	return detection_outputs;
}

Variable YOLOv5::forward(const std::vector<Variable>& xs) {
	auto outputs = forward_detect(xs[0]);
	return outputs[0];
}

void YOLOv5::load_weights(const std::string& weights_path) {
	cnpy::npz_t npz = cnpy::npz_load(weights_path);

	// Backbone
	backbone_0->load_from_npz(npz, "model.0");
	backbone_1->load_from_npz(npz, "model.1");
	backbone_2->load_from_npz(npz, "model.2");
	backbone_3->load_from_npz(npz, "model.3");
	backbone_4->load_from_npz(npz, "model.4");
	backbone_5->load_from_npz(npz, "model.5");
	backbone_6->load_from_npz(npz, "model.6");
	backbone_7->load_from_npz(npz, "model.7");
	backbone_8->load_from_npz(npz, "model.8");
	backbone_9->load_from_npz(npz, "model.9");

	// Neck
	neck_10->load_from_npz(npz, "model.10");
	neck_13->load_from_npz(npz, "model.13");
	neck_14->load_from_npz(npz, "model.14");
	neck_17->load_from_npz(npz, "model.17");
	neck_18->load_from_npz(npz, "model.18");
	neck_20->load_from_npz(npz, "model.20");
	neck_21->load_from_npz(npz, "model.21");
	neck_23->load_from_npz(npz, "model.23");

	// Detection heads
	detect_p3->load_params_from_npz(npz, "model.24.m.0");
	detect_p4->load_params_from_npz(npz, "model.24.m.1");
	detect_p5->load_params_from_npz(npz, "model.24.m.2");
}
