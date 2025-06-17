#include "dataset/dataloader.hpp"

#include "container/tensor/tensor_all.hpp"

using namespace tensor;

DataLoader::DataLoader(Dataset dataset,
						size_t batch_size,
						bool shuffle)
		: 	dataset(dataset), 
			batch_size(batch_size), 
			shuffle(shuffle) {
	data_size = dataset.size();

	this->reset();
}

void DataLoader::reset() {
	current_index = 0;
	if (shuffle)
		indices = permutation(data_size);
	else
		indices = arrange<size_t>(data_size);
}

bool DataLoader::has_next() const {
	return current_index < dataset.size();
}

std::pair<Tensor<>, Tensor<>> DataLoader::next() {
	std::vector<Tensor<>> batch_x;
	std::vector<Tensor<>> batch_y;

	for (size_t i = 0; i < batch_size && has_next(); i++, current_index++) {
		size_t idx = indices({current_index});
		batch_x.push_back(dataset.get_data(idx));
		batch_y.push_back(dataset.get_label(idx));
	}

	return { stack(batch_x), stack(batch_y) };	
}
