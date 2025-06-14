#include "include/dataset/dataset.hpp"
#include "include/container/tensor/tensor_all.hpp"

#include <cmath>
#include <random>

SpiralDataset::SpiralDataset(size_t num_data,
							size_t num_class,
							bool train)
		: Dataset(train),
		num_data(num_data),
		num_class(num_class) {
	init_dataset();
}

void SpiralDataset::init_dataset() {
	const size_t input_dim = 2;
	const size_t total = num_data * num_class;
	const size_t seed = train ? 1984 : 2020;

	Tensor<> x_data({total, input_dim});
	Tensor<> t_data({total});
	Tensor<> noise = randn(total, 1, seed);

	for (size_t j = 0; j < num_class; j++) {
		for (size_t i = 0; i < num_data; i++) {
			float rate = static_cast<float>(i) / num_data;
			float r = 1.0f * rate;
			size_t ix = j * num_data + i;

			float theta = j * 4.0f + 4.0f * rate + noise({ix, 0}) * 0.2f;

			x_data({ix, 0}) = r * std::sin(theta);
			x_data({ix, 1}) = r * std::cos(theta);
			t_data({ix}) = static_cast<float>(j);
		}
	}

	Tensor<size_t> indices = permutation(total);

	Tensor<> x_shuffled({total, input_dim});
	Tensor<> t_shuffled({total});

	for (size_t i = 0; i < total; i++) {
		for (size_t d = 0; d < input_dim; d++)
			x_shuffled({i, d}) = x_data({indices({i}), d});
		t_shuffled({i}) = t_data({indices({i})});
	}

	data = x_shuffled;
	label = t_shuffled;
}

