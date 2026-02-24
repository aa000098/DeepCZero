#pragma once

#include "container/layer/layer.hpp"
#include "container/layer/model.hpp"

#include <vector>
#include <memory>
#include <string>
#include <cmath>
#include <algorithm>

namespace cnpy {
	struct NpyArray;
	using npz_t = std::map<std::string, NpyArray>;
}

namespace layer {

// Conv-BatchNorm-SiLU block
class CBS : public Layer {
private:
	std::shared_ptr<Conv2d> conv;
	std::shared_ptr<BatchNorm2d> bn;

public:
	CBS() = default;
	CBS(size_t in_channels, size_t out_channels,
		std::pair<size_t, size_t> kernel_size = {1, 1},
		std::pair<size_t, size_t> stride = {1, 1},
		std::pair<size_t, size_t> pad = {0, 0});

	Variable forward(const std::vector<Variable>& xs) override;
	void load_from_npz(const cnpy::npz_t& npz, const std::string& prefix);
};

// Bottleneck residual block
class Bottleneck : public Layer {
private:
	std::shared_ptr<CBS> cv1;
	std::shared_ptr<CBS> cv2;
	bool shortcut;

public:
	Bottleneck() = default;
	Bottleneck(size_t in_channels, size_t out_channels,
			   bool shortcut = true, float expansion = 0.5f);

	Variable forward(const std::vector<Variable>& xs) override;
	void load_from_npz(const cnpy::npz_t& npz, const std::string& prefix);
};

// C3: CSP Bottleneck with 3 convolutions
class C3 : public Layer {
private:
	std::shared_ptr<CBS> cv1;
	std::shared_ptr<CBS> cv2;
	std::shared_ptr<CBS> cv3;
	std::vector<std::shared_ptr<Bottleneck>> bottlenecks;
	size_t num_bottlenecks;

public:
	C3() = default;
	C3(size_t in_channels, size_t out_channels,
	   size_t n = 1, bool shortcut = true, float expansion = 0.5f);

	Variable forward(const std::vector<Variable>& xs) override;
	void load_from_npz(const cnpy::npz_t& npz, const std::string& prefix);
};

// SPPF: Spatial Pyramid Pooling - Fast
class SPPF : public Layer {
private:
	std::shared_ptr<CBS> cv1;
	std::shared_ptr<CBS> cv2;
	std::pair<size_t, size_t> pool_size;

public:
	SPPF() = default;
	SPPF(size_t in_channels, size_t out_channels, size_t k = 5);

	Variable forward(const std::vector<Variable>& xs) override;
	void load_from_npz(const cnpy::npz_t& npz, const std::string& prefix);
};

} // namespace layer


// YOLOv5 full model (outside namespace layer, like VGG16)
class YOLOv5 : public Model {
private:
	static constexpr size_t NUM_ANCHORS = 3;

	size_t num_classes;
	float depth_multiple;
	float width_multiple;

	// Helper functions
	static size_t make_divisible(size_t v, size_t divisor);
	size_t width(size_t channels) const;
	size_t depth(size_t n) const;

	// Backbone
	std::shared_ptr<layer::CBS>  backbone_0;  // 3 -> w(64),  k=6, s=2, p=2
	std::shared_ptr<layer::CBS>  backbone_1;  // w(64) -> w(128), k=3, s=2, p=1
	std::shared_ptr<layer::C3>   backbone_2;  // w(128), n=d(3)
	std::shared_ptr<layer::CBS>  backbone_3;  // w(128) -> w(256), k=3, s=2, p=1
	std::shared_ptr<layer::C3>   backbone_4;  // w(256), n=d(6)
	std::shared_ptr<layer::CBS>  backbone_5;  // w(256) -> w(512), k=3, s=2, p=1
	std::shared_ptr<layer::C3>   backbone_6;  // w(512), n=d(9)
	std::shared_ptr<layer::CBS>  backbone_7;  // w(512) -> w(1024), k=3, s=2, p=1
	std::shared_ptr<layer::C3>   backbone_8;  // w(1024), n=d(3)
	std::shared_ptr<layer::SPPF> backbone_9;  // w(1024), k=5

	// Neck - top-down
	std::shared_ptr<layer::CBS> neck_10;  // w(1024) -> w(512), 1x1
	// 11: upsample (no layer)
	// 12: concat with backbone_6 output
	std::shared_ptr<layer::C3>  neck_13;  // w(1024) -> w(512), shortcut=false
	std::shared_ptr<layer::CBS> neck_14;  // w(512) -> w(256), 1x1
	// 15: upsample (no layer)
	// 16: concat with backbone_4 output
	std::shared_ptr<layer::C3>  neck_17;  // w(512) -> w(256), shortcut=false

	// Neck - bottom-up
	std::shared_ptr<layer::CBS> neck_18;  // w(256) -> w(256), k=3, s=2
	// 19: concat with neck_14 output
	std::shared_ptr<layer::C3>  neck_20;  // w(512) -> w(512), shortcut=false
	std::shared_ptr<layer::CBS> neck_21;  // w(512) -> w(512), k=3, s=2
	// 22: concat with neck_10 output
	std::shared_ptr<layer::C3>  neck_23;  // w(1024) -> w(1024), shortcut=false

	// Detection heads
	std::shared_ptr<layer::Conv2d> detect_p3;  // w(256) -> out_ch, 1x1
	std::shared_ptr<layer::Conv2d> detect_p4;  // w(512) -> out_ch, 1x1
	std::shared_ptr<layer::Conv2d> detect_p5;  // w(1024) -> out_ch, 1x1

	// Multi-scale outputs stored after forward
	std::vector<Variable> detection_outputs;

public:
	YOLOv5(size_t num_classes = 80,
		   float depth_multiple = 0.33f,
		   float width_multiple = 0.50f,
		   bool pretrained = false);

	// Layer interface (returns first detection output)
	Variable forward(const std::vector<Variable>& xs) override;

	// Primary API: returns {P3, P4, P5} detection outputs
	std::vector<Variable> forward_detect(const Variable& x);

	void load_weights(const std::string& weights_path);

	const std::vector<Variable>& get_detection_outputs() const {
		return detection_outputs;
	}
};
