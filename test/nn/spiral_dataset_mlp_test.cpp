#include "deepczero.hpp"

#include <cassert>

void test_spiral_dataset_mlp() {

	size_t max_epoch = 300;
	size_t batch_size = 30;
	size_t hidden_size = 10;
	float lr = 1.0f;

	size_t num_data = 100;
	size_t num_class = 3;

	SpiralDataset train_ds(num_data, num_class, true);
	Variable train_X = train_ds.get_data();
	Variable train_t = train_ds.get_target();

	MLP model({hidden_size, 3});
	SGD optimizer(lr);
	optimizer.setup(model);

	size_t data_size = train_X.size();
	size_t max_iter = data_size / batch_size;

	for (size_t epoch = 0; epoch < max_epoch; epoch++) {
		Tensor<> index = permutation(data_size);
		size_t sum_loss = 0;
		
	}

}

int main() {

}
