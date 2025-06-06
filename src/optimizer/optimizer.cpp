#include "optimizer/optimizer.hpp"

void Optimizer::setup(const Model target) {
	this->target = target;
}

void Optimizer::update() {
	std::vector<Parameter> params = target.get_params();

	// TODO: preprocessing with hooks
	/*
	 * for (auto& h : hooks)
	 * 		h(params)
	*/

	for (auto& param : params)
		update_one(param);
}

/*
 * void add_hook(Hook h) {
 * 		hooks.push_back(h);
 * }
 */ 

void SGD::update_one(Parameter param) {
	param.data() -= lr * param.grad().data();
}
