#include "deepczero.hpp"

#include <iostream>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <vector>

void test_upsample_forward() {
	std::cout << "=== Upsample Forward ===" << std::endl;

	// [1, 1, 2, 2] -> [1, 1, 4, 4] with scale=2
	Tensor<> x_data({1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
	Variable x(x_data);

	Variable y = upsample(x, 2);

	auto shape = y.shape();
	std::cout << "Input shape: [1,1,2,2] -> Output shape: ["
			  << shape[0] << "," << shape[1] << ","
			  << shape[2] << "," << shape[3] << "]" << std::endl;

	assert(shape[0] == 1);
	assert(shape[1] == 1);
	assert(shape[2] == 4);
	assert(shape[3] == 4);

	// Each pixel is repeated 2x2
	assert(y.data()({0, 0, 0, 0}) == 1.0f);
	assert(y.data()({0, 0, 0, 1}) == 1.0f);
	assert(y.data()({0, 0, 1, 0}) == 1.0f);
	assert(y.data()({0, 0, 1, 1}) == 1.0f);
	assert(y.data()({0, 0, 0, 2}) == 2.0f);
	assert(y.data()({0, 0, 2, 0}) == 3.0f);
	assert(y.data()({0, 0, 2, 2}) == 4.0f);

	std::cout << "Upsample forward test passed!" << std::endl;
}

void test_upsample_backward() {
	std::cout << "=== Upsample Backward ===" << std::endl;

	Tensor<> x_data({1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
	Variable x(x_data);

	Variable y = upsample(x, 2);
	Variable loss = sum(y);
	loss.backward();

	// Each input pixel contributes to scale^2 = 4 output pixels
	// So gradient should be 4.0 for each input pixel
	assert(x.has_grad());
	assert(x.grad().data()({0, 0, 0, 0}) == 4.0f);
	assert(x.grad().data()({0, 0, 1, 1}) == 4.0f);

	std::cout << "Upsample backward test passed!" << std::endl;
}

void test_upsample_bilinear() {
	std::cout << "=== Upsample Bilinear (vs PyTorch interpolate) ===" << std::endl;

	struct Case {
		bool align_corners;
		std::vector<float> y;
		std::vector<float> gx;
	};
	std::vector<Case> cases = {
		{false,
		 {1.0f, 1.25f, 1.75f, 2.25f, 2.75f, 3.0f, 1.75f, 2.0f, 2.5f, 3.0f, 3.5f, 3.75f,
		  3.25f, 3.5f, 4.0f, 4.5f, 5.0f, 5.25f, 4.0f, 4.25f, 4.75f, 5.25f, 5.75f, 6.0f},
		 {4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f}},
		{true,
		 {1.0f, 1.4f, 1.8f, 2.2f, 2.6f, 3.0f, 2.0f, 2.4f, 2.8f, 3.2f, 3.6f, 4.0f,
		  3.0f, 3.4f, 3.8f, 4.2f, 4.6f, 5.0f, 4.0f, 4.4f, 4.8f, 5.2f, 5.6f, 6.0f},
		 {3.6f, 4.8f, 3.6f, 3.6f, 4.8f, 3.6f}},
	};

	for (const auto& c : cases) {
		Tensor<> x_data({1, 1, 2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
		Variable x(x_data);

		Variable y = upsample(x, 2, "bilinear", c.align_corners);
		assert((y.shape() == std::vector<size_t>{1, 1, 4, 6}));
		Tensor<> y_data = y.data().contiguous();
		for (size_t i = 0; i < c.y.size(); i++)
			assert(std::abs(y_data.raw_data()[i] - c.y[i]) < 1e-5f);

		Variable loss = sum(y);
		loss.backward();
		Tensor<> gx = x.grad().data().contiguous();
		for (size_t i = 0; i < c.gx.size(); i++)
			assert(std::abs(gx.raw_data()[i] - c.gx[i]) < 1e-5f);

		std::cout << "align_corners=" << (c.align_corners ? "true" : "false")
				  << " matches PyTorch" << std::endl;
	}

	bool threw = false;
	try {
		upsample(Variable(Tensor<>({1, 1, 2, 2}, 1.0f)), 2, "bicubic");
	} catch (const std::invalid_argument&) {
		threw = true;
	}
	assert(threw);

	std::cout << "Upsample bilinear test passed!" << std::endl;
}

int main() {
	test_upsample_forward();
	test_upsample_backward();
	test_upsample_bilinear();
	std::cout << "\nAll Upsample tests passed!" << std::endl;
	return 0;
}
