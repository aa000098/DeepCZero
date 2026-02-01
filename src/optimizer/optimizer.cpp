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
	if (!param.has_grad()) return;
	param.data() -= lr * param.grad().data();
}

void MomentumSGD::update_one(Parameter param) {
	if (!param.has_grad()) return;

	const std::uintptr_t v_key = param.id();

	if (vs.find(v_key) == vs.end())
		vs[v_key] = Tensor(param.shape(), 0.0f);

	vs[v_key] = vs[v_key] * momentum - lr * param.grad().data();
	param.data() += vs[v_key];
}
