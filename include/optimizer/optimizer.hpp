#pragma once

#include "container/layer/layer.hpp"
#include "container/layer/model.hpp"
#include "container/parameter.hpp"

#include <memory>
#include <unordered_map>

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

class MomentumSGD : public Optimizer {
private:
	float lr;
	float momentum;
	std::unordered_map<std::uintptr_t, Tensor<float>> vs;

public:
	MomentumSGD(float lr = 0.01,
				float momentum = 0.9)
		: lr(lr), momentum(momentum) {};

	void update_one(Parameter param);

};


