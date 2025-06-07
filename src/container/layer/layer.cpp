#include "container/layer/layer.hpp"
#include "function/function.hpp"
#include "function/activation_functions.hpp"

using function::Sigmoid;

namespace layer {
	Parameter Layer::get_param(const std::string& name) const {
		auto it = params.find(name);
		if (it != params.end()) return it->second;
		else throw std::runtime_error("Parameter not found: " + name);
	}

	std::shared_ptr<Layer> Layer::get_sublayer(const std::string& name) const {
		auto it = sublayers.find(name);
		if (it != sublayers.end()) return it->second;
		else throw std::runtime_error("Sublayer not found : " + name);
	}

	Variable Layer::operator()(const std::vector<Variable>& inputs) {
		this->inputs = inputs;
		this->output = forward(inputs);
		return this->output;
	}

	Variable Layer::operator()(const std::initializer_list<Variable>& inputs) {
		std::vector<Variable> input_vec(inputs);
		return (*this)(input_vec);
	}


	Variable Layer::operator()(const Variable& input) {
		return (*this)({input});
	}

	void Layer::cleargrads() {
		for (auto& pair : get_params())
			pair.cleargrad();
	}

	std::vector<Parameter> Layer::get_params() {
		std::vector<Parameter> all_params;

		for (auto& pair : params)
			all_params.push_back(pair.second);

		for (auto& pair : sublayers) {
			std::vector<Parameter> child_params = pair.second->get_params();
			all_params.insert(all_params.end(), child_params.begin(), child_params.end());
		}

		return all_params;
	}

	
	Linear::Linear( size_t out_size, 
					bool nobias,
					/*dtype = float32, */
					size_t in_size) 
			: out_size(out_size), in_size(in_size) {
		Parameter W({}, "W");
		register_params("W", W);
		if (nobias) {
			Parameter b({}, "b");
			register_params("b", b);
		} else {
			Tensor b_data(out_size);
			Parameter b(b_data, "b");
			register_params("b", b);
		}
	};
		

	void Linear::init_W() {
		Tensor W_data = randn(in_size, out_size); 
		W_data *= std::sqrt(1/in_size);
		params["W"].data() = W_data;
	}

	Variable Linear::forward(const std::vector<Variable>& xs) {
		const Variable& x = xs[0];
		const Parameter& W = get_param("W");
		const Parameter& b = get_param("b");
			if (W.data().empty()) {
			in_size = x.shape()[1];
			init_W();
		}
		
		Variable y = linear(x, W, b);
		return y;
	}

}
