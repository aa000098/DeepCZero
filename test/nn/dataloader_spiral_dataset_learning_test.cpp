#include "deepczero.hpp"

#include <iostream>


void test_dataloader_spiral_dataset_learning() {
    size_t num_data = 100;
    size_t num_class = 3;
    size_t batch_size = 30;

    SpiralDataset train_set(num_data, num_class, true);
    SpiralDataset test_set(num_data, num_class, false);

    DataLoader train_loader(train_set, batch_size, true);
    DataLoader test_loader(test_set, batch_size, false);

	size_t max_epoch = 300;
	size_t hidden_size = 10;
	float lr = 1.0f;

	MLP model({hidden_size, 3});
	SGD optimizer(lr);
	optimizer.setup(model);

	for (size_t epoch = 0; epoch < max_epoch; epoch++) {
		//train_loader.reset();
		//test_loader.reset();
		float sum_loss = 0, sum_acc = 0;
			//while(train_loader.has_next()) {
			for (const auto& [x, t] : train_loader) {
				//auto [x, t] = train_loader.next();

				Variable y = model(x);
				Variable loss = softmax_cross_entropy_error(y, t);
				Variable acc = accuracy(y, t);

				model.cleargrads();
				loss.backward();
				optimizer.update();

				sum_loss += loss({0}) * static_cast<float>(t.size());
				sum_acc += acc({0}) * static_cast<float>(t.size());
			}
			std::cout << "epoch : " << epoch+1 << std::endl;
			std::cout << "train loss : " << sum_loss / static_cast<float>(train_set.size()) << ", accuray : " << sum_acc / static_cast<float>(train_set.size()) << std::endl;

			sum_loss = 0;
			sum_acc = 0;
			{
				dcz::UsingConfig no_grad(true);
				
				for (const auto& [x, t] : test_loader) {
					Variable y = model(x);
					Variable loss = softmax_cross_entropy_error(y, t);
					Variable acc = accuracy(y, t);

					sum_loss += loss({0}) * static_cast<float>(t.size());
					sum_acc += acc({0}) * static_cast<float>(t.size());
				}
			}
			std::cout << "test loss : " << sum_loss / static_cast<float>(test_set.size()) << ", accuray : " << sum_acc / static_cast<float>(test_set.size()) << std::endl;

	}

}

int main() {
	test_dataloader_spiral_dataset_learning();
}
