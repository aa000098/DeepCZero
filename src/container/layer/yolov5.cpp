#include "container/layer/yolov5.hpp"
#include "function/ops/ops_all.hpp"
#include "container/variable_ops.hpp"

namespace layer {

// ===== CBS (Conv-BatchNorm-SiLU) =====

CBS::CBS(size_t in_channels, size_t out_channels,
		 std::pair<size_t, size_t> kernel_size,
		 std::pair<size_t, size_t> stride,
		 std::pair<size_t, size_t> pad) {
	conv = std::make_shared<Conv2d>(out_channels, kernel_size, stride, pad,
									/*no_bias=*/true, in_channels);
	bn = std::make_shared<BatchNorm2d>(out_channels);

	register_sublayers("conv", conv);
	register_sublayers("bn", bn);
}

Variable CBS::forward(const std::vector<Variable>& xs) {
	Variable y = (*conv)(xs[0]);
	y = (*bn)(y);
	y = silu(y);
	return y;
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

} // namespace layer
