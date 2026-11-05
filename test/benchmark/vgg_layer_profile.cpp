// Per-operation profile of a VGG16-shaped forward pass (batch 1, 224x224).
// Breaks each conv layer into pad / im2col / GEMM / bias-add and times ReLU,
// pooling and the FC layers separately, so the bottleneck is visible without
// a sampling profiler.
#include "deepczero.hpp"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>

using Clock = std::chrono::high_resolution_clock;

static double ms_since(Clock::time_point t0) {
	return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

static Tensor<> random_tensor(const std::vector<size_t>& shape, float scale = 0.1f) {
	Tensor<> t(shape);
	for (auto& v : t.raw_data()) v = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * scale;
	return t;
}

struct Totals {
	std::map<std::string, double> ms;
	void add(const std::string& k, double v) { ms[k] += v; }
};

struct ConvCfg { size_t cin, cout, hw; };

int main() {
	srand(0);
	dcz::UsingConfig eval_mode("train", false);

	const std::pair<size_t, size_t> k{3, 3}, s{1, 1}, p{1, 1};
	// VGG16 conv stack, batch 1
	std::vector<ConvCfg> convs = {
		{3, 64, 224}, {64, 64, 224},
		{64, 128, 112}, {128, 128, 112},
		{128, 256, 56}, {256, 256, 56}, {256, 256, 56},
		{256, 512, 28}, {512, 512, 28}, {512, 512, 28},
		{512, 512, 14}, {512, 512, 14}, {512, 512, 14},
	};
	std::vector<size_t> pool_after = {1, 3, 6, 9, 12};  // conv index after which a 2x2 pool runs

	Totals tot;
	std::cout << std::left << std::setw(16) << "layer"
	          << std::right << std::setw(10) << "pad"
	          << std::setw(10) << "im2col"
	          << std::setw(10) << "gemm"
	          << std::setw(10) << "bias"
	          << std::setw(10) << "relu"
	          << std::setw(10) << "pool"
	          << std::setw(12) << "conv2d()" << std::endl;

	Tensor<> x = random_tensor({1, 3, 224, 224}, 1.0f);
	for (size_t li = 0; li < convs.size(); ++li) {
		const ConvCfg& c = convs[li];
		Tensor<> W = random_tensor({c.cout, c.cin, 3, 3});
		Tensor<> b = random_tensor({c.cout});

		// --- decomposed conv ---
		auto t0 = Clock::now();
		Tensor<> padded = x.pad({{0, 0}, {0, 0}, {1, 1}, {1, 1}}, 0.0f);
		double t_pad = ms_since(t0);

		t0 = Clock::now();
		Tensor<> col = tensor::im2col_array(x, k, s, p, false);  // includes its own pad
		double t_im2col = ms_since(t0) - t_pad;  // im2col minus the pad it does internally
		if (t_im2col < 0) t_im2col = 0;

		const auto cs = col.get_shape();
		size_t CKK = cs[1] * 3 * 3, OH = cs[4], OW = cs[5];
		t0 = Clock::now();
		Tensor<> y = tensor::dot(W.reshape({c.cout, CKK}), col.reshape({1, CKK, OH * OW})).reshape({1, c.cout, OH, OW});
		double t_gemm = ms_since(t0);

		t0 = Clock::now();
		y += b.reshape({1, c.cout, 1, 1});
		double t_bias = ms_since(t0);

		// --- whole conv2d() op through the Variable API, for cross-checking ---
		t0 = Clock::now();
		Variable yv = conv2d(Variable(x), Variable(W), Variable(b), s, p);
		double t_conv = ms_since(t0);

		t0 = Clock::now();
		Variable r = relu(yv);
		double t_relu = ms_since(t0);

		double t_pool = 0;
		if (std::find(pool_after.begin(), pool_after.end(), li) != pool_after.end()) {
			t0 = Clock::now();
			r = pooling(r, {2, 2}, {2, 2});
			t_pool = ms_since(t0);
		}

		std::cout << std::left << std::setw(16) << ("conv" + std::to_string(li) + " " + std::to_string(c.cin) + "->" + std::to_string(c.cout))
		          << std::right << std::fixed << std::setprecision(1)
		          << std::setw(10) << t_pad << std::setw(10) << t_im2col << std::setw(10) << t_gemm
		          << std::setw(10) << t_bias << std::setw(10) << t_relu << std::setw(10) << t_pool
		          << std::setw(12) << t_conv << std::endl;

		tot.add("pad", t_pad); tot.add("im2col", t_im2col); tot.add("gemm", t_gemm);
		tot.add("bias", t_bias); tot.add("relu", t_relu); tot.add("pool", t_pool);
		tot.add("conv2d() total", t_conv);

		x = r.data();
	}

	// FC layers
	Tensor<> flat = x.reshape({1, 512 * 7 * 7});
	std::vector<std::pair<size_t, size_t>> fcs = {{25088, 4096}, {4096, 4096}, {4096, 1000}};
	double t_fc = 0;
	for (auto [in, out] : fcs) {
		Tensor<> W = random_tensor({in, out});
		Tensor<> b = random_tensor({out});
		auto t0 = Clock::now();
		Variable h = linear(Variable(flat), Variable(W), Variable(b));
		t_fc += ms_since(t0);
		flat = h.data();
	}
	tot.add("fc(3 layers)", t_fc);

	std::cout << "\nTotals (ms, batch 1):" << std::endl;
	for (auto& [k2, v] : tot.ms)
		std::cout << "  " << std::left << std::setw(16) << k2 << std::right << std::setw(10) << std::fixed << std::setprecision(1) << v << std::endl;
	std::cout << "  (decomposed conv = pad+im2col+gemm+bias; conv2d() is the real op measured separately)" << std::endl;
	return 0;
}
