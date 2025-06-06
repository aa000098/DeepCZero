#pragma once

#include "container/layer/layer.hpp"
#include "container/layer/model.hpp"
#include "container/parameter.hpp"

#include <memory>

using layer::Layer;

class Optimizer {
private:
	Model target;
	// TODO: std::vector<Hook> add_hook (preprocessor)
 
public:
	Optimizer() = default;

	void setup(const Model target);
	void update();
	virtual void update_one(Parameter Param) = 0;
	// void add_hook(Hook h);

};

class SGD : public Optimizer {
private:
	float lr;

public:
	SGD(float lr = 0.001) : lr(lr) {};

	void update_one(Parameter Param);

};
