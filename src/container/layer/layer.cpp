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


	std::unordered_map<std::string, Parameter> Layer::flatten_params(const std::string& parent_key) {
		std::unordered_map<std::string, Parameter> params_dict;

		for (const auto& [name, param] : this->params) {
			std::string key = parent_key.empty() ? name : parent_key + "/" + name;
			params_dict[key] = param;
		}

		for (const auto& [name, sublayer] : this->sublayers) {
			std::string key = parent_key.empty() ? name : parent_key + "/" + name;
			auto sub_params = sublayer->flatten_params(key);
			params_dict.insert(sub_params.begin(), sub_params.end());
		}
		return params_dict;
	}


	void Layer::save_weights(std::string path) {
		// to_cpu();
		std::unordered_map<std::string, Parameter> params_dict = flatten_params();

		std::vector<Parameter> params_array = get_params();
		
	}

	void Layer::load_weights(std::string path) {
		std::ifstream fin(path, std::ios::binary);

		if (!fin.is_open())
			throw std::runtime_error("file not opened");


//		fin.read((char *)& 

		std::unordered_map<std::string, Parameter> params_dict = flatten_params();
		for (const auto& [name, param] : params_dict)
			params[name] = param;
	
	}
	
	
// [Linear]
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
		W_data *= std::sqrt(1/static_cast<float>(in_size));
		params["W"].data() = W_data;
	}

	Variable Linear::forward(const std::vector<Variable>& xs) {
		const Variable& x = xs[0];
		const Parameter& W = get_param("W");
		const Parameter& b = get_param("b");
		if (W.data().empty()) {
			in_size = x.shape().back();
			init_W();
		}
		
		Variable y = linear(x, W, b);
		return y;
	}

}
