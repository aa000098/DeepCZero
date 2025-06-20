#pragma once

#include "dataset/dataset.hpp"
#include "dataset/iterator.hpp"

using tensor::Tensor;

class DataLoader {
private:
	Dataset dataset;
	size_t batch_size;
	bool shuffle;
	
	size_t data_size;

	size_t current_index;
	Tensor<size_t> indices;

public:
	DataLoader(Dataset dataset,
				size_t batch_size,
				bool shuffle = true);

	void reset();
	bool has_next() const;
	std::pair<Tensor<>, Tensor<>> next_batch();

	size_t size() { return data_size; };

public:
	std::pair<Tensor<>, Tensor<>> operator[](size_t index);
	Iterator<DataLoader> begin();
	Iterator<DataLoader> end();
};
