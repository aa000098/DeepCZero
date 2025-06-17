#pragma once

#include "container/tensor/tensor_all.hpp"
#include "dataset/transform/transform_all.hpp"

#include <string>
#include <memory>

using tensor::Tensor;

class Dataset {
protected:
	Tensor<> data;
	Tensor<> label;

	bool train;

	std::shared_ptr<Transform<float>> transform;
	std::shared_ptr<Transform<float>> target_transform;

public:
	Dataset(bool train = true,
			std::shared_ptr<Transform<float>> transform = nullptr,
			std::shared_ptr<Transform<float>> target_transform = nullptr) 
		: train(train), transform(transform), target_transform(target_transform) {};

	Tensor<> get_data() const { return data; };
	Tensor<> get_data(size_t index) const;
	Tensor<> get_label() const { return label; };
	Tensor<> get_label(size_t index) const;
	size_t size() const { return data.get_shape()[0]; };
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

	Tensor<> get_data(size_t index);
	Tensor<> get_label(size_t index);

};
