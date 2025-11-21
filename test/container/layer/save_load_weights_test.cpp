#include "deepczero.hpp"

#include <iostream>
#include <chrono>
#include <omp.h>

void test_mnist_dataset_learning() {
    
    size_t batch_size = 1000;

   	MNISTDataset train_set(true);
    MNISTDataset test_set(false);

    DataLoader train_loader(train_set, batch_size);
    DataLoader test_loader(test_set, batch_size, false);

	size_t max_epoch = 1;
	size_t hidden_size = 3;
	

	MLP model({hidden_size, 10});
	SGD optimizer;
	optimizer.setup(model);
	
	std::string weight_name = "my_mlp.dcz";
	std::filesystem::path weight_path = get_cache_file_path(weight_name);

    // 이미 있으면 그대로 반환
    if (std::filesystem::exists(weight_path)) {
        model.load_weights(weight_path);
		std::cout << "[Weights Loaded]" << std::endl;
    }

	for (size_t epoch = 0; epoch < max_epoch; epoch++) {
		float sum_loss = 0, sum_acc = 0;
		size_t batch_index = 1;
		size_t total_batches = (train_set.size() + batch_size - 1) / batch_size;

			for (const auto& [x, t] : train_loader) {
				auto start = std::chrono::high_resolution_clock::now();

    			std::cout << "Epoch " << epoch+1 << ", Batch " << batch_index++ << "/" << total_batches << std::endl;

	
				Variable y = model(x);
				Variable loss = softmax_cross_entropy_error(y, t);
				Variable acc = accuracy(y, t);

				model.cleargrads();
				loss.backward();
				optimizer.update();

				sum_loss += loss({0}) * static_cast<float>(t.size());
				sum_acc += acc({0}) * static_cast<float>(t.size());

				auto end = std::chrono::high_resolution_clock::now();
				std::chrono::duration<double> elapsed = end - start;
				std::cout << " -> Batch Time: " << elapsed.count() << " seconds\n";
			}
			std::cout << "epoch : " << epoch+1 << std::endl;
			std::cout << "train loss : " << sum_loss / static_cast<float>(train_set.size()) << ", accuray : " << sum_acc / static_cast<float>(train_set.size()) << std::endl;

			sum_loss = 0;
			sum_acc = 0;
			{
				dcz::UsingConfig no_grad("enable_backprop", true);
				
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
	model.save_weights(weight_name);
	std::cout << "[Weights Saved]" << std::endl;
}


int main() {
	omp_set_num_threads(16);
	test_mnist_dataset_learning();
}
