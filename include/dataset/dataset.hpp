#pragma once

#include "include/container/tensor/tensor_all.hpp"

#include <string>

class Dataset {
protected:
	Tensor<> data;
	Tensor<> label;
	bool train;

public:
	Dataset(bool train = true) : train(train) {};

	Tensor<> get_data() { return data; };
	Tensor<> get_label() { return label; };
};

class SpiralDataset : public Dataset {
private:
	size_t num_data;
	size_t num_class;

public:
	SpiralDataset(	size_t num_data, 
					size_t num_class,
					bool train = true);

	void init_dataset();

};

class BigDataset : public Dataset {
private:
	size_t num_data;
	size_t num_class;

public:
	BigDataset(	size_t num_data,
				size_t num_class,
				bool train = true);

	Tensor<> get_data();
	Tensor<> get_label();

};
