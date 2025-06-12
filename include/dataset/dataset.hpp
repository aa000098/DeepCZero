#pragma once

#include "include/container/variable.hpp"

#include <string>

class Dataset {
protected:
	Variable data;
	Variable target;
	bool train;

public:
	Dataset(bool train = true) : train(train) {};

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

	Variable get_data() { return data; };
	Variable get_target() { return target; };
};
