#include "deepczero.hpp"

#include <cassert>
#include <iostream>

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

	size_t data_size = train_X.shape()[0];
	size_t max_iter = data_size / batch_size;

	for (size_t epoch = 0; epoch < max_epoch; epoch++) {
		Tensor<size_t> index = permutation(data_size);
		size_t sum_loss = 0;

		for (size_t i = 0; i < max_iter; i++) {
			size_t start = i * batch_size;
			size_t end = std::min(start + batch_size, data_size);

			Tensor<size_t> batch_index = index.slice(0, start, end);

			Variable batch_x = train_X.gather_rows(batch_index);
			Variable batch_t = train_t.gather_rows(batch_index);
			Variable y = model(batch_x);
			Variable loss = softmax_cross_entropy_error(y, batch_t);

			model.cleargrads();
			loss.backward();
			optimizer.update();

			sum_loss += loss({0}) * static_cast<float>(batch_t.size());
		
		}

		float avg_loss = sum_loss / static_cast<float>(data_size);
		std::cout << "epoch : " << epoch+1 << ", loss : " << avg_loss << std::endl; 
		
	}

}

int main() {
	test_spiral_dataset_mlp();
}
