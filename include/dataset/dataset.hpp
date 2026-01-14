#pragma once

#include "container/tensor/tensor_all.hpp"
#include "dataset/transform/transform_all.hpp"

#include <string>
#include <memory>
#include <map>

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

	Tensor<> get_data() const;
	Tensor<> get_data(size_t index) const;
	Tensor<> get_label() const;
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
	BigDataset(	size_t num_data = 100,
				size_t num_class = 3,
				bool train = true);

	Tensor<> get_data(size_t index);
	Tensor<> get_label(size_t index);

};

class MNISTDataset : public Dataset {
protected:
	Tensor<> _load_data(std::string& file_path);
	Tensor<> _load_label(std::string& file_path);

public:
	MNISTDataset(bool train = true);
	void init_dataset();
	void show(size_t index = 0);
};

class ImageNetDataset : public Dataset {
private:
	std::map<int, std::string> labels_map;

public:
	ImageNetDataset(bool train = true);
	void init_dataset();
	std::string labels(int class_id) const;
};