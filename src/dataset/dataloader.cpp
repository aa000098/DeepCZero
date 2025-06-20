#include "dataset/dataloader.hpp"
#include "dataset/iterator.hpp"

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

std::pair<Tensor<>, Tensor<>> DataLoader::next_batch() {
	std::vector<Tensor<>> batch_y;
	std::vector<Tensor<>> batch_t;

	for (size_t i = 0; i < batch_size && has_next(); i++, current_index++) {
		size_t idx = indices({current_index});
		batch_y.push_back(dataset.get_data(idx));
		batch_t.push_back(dataset.get_label(idx));
	}

	return { stack(batch_y), stack(batch_t) };	
}


// [Iterator Base]
std::pair<Tensor<>, Tensor<>> DataLoader::operator[](size_t index) {
	size_t start = index * batch_size;
	size_t end = std::min(start + batch_size, data_size);

	std::vector<Tensor<>> batch_y;
	std::vector<Tensor<>> batch_t;

	for (size_t i = start; i < end; i++) {
		size_t idx = indices({i});
		batch_y.push_back(dataset.get_data(idx));
		batch_t.push_back(dataset.get_label(idx));
	}
	return { stack(batch_y), stack(batch_t) };
}
	
Iterator<DataLoader> DataLoader::begin() {
	reset();
	return Iterator<DataLoader>(this, 0);
}

Iterator<DataLoader> DataLoader::end() {
	size_t num_batches = (data_size + batch_size - 1) / batch_size;
	return Iterator<DataLoader>(this, num_batches);
}

