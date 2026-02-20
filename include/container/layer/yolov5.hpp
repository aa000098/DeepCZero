#pragma once

#include "container/layer/layer.hpp"

#include <vector>
#include <memory>

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
};

} // namespace layer
