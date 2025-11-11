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
		this->inputs = {inputs};
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

/*
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
*/	
	
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


	Conv2d::Conv2d(size_t out_channels,
					std::pair<size_t, size_t> kernel_size,
					std::pair<size_t, size_t> stride,
					std::pair<size_t, size_t> pad,
					bool no_bias,
					size_t in_channels) 
		: out_channels(out_channels), kernel_size(kernel_size), stride(stride), pad(pad), no_bias(no_bias), in_channels(in_channels) {
		Parameter W({}, "W");
		register_params("W", W);
		if (in_channels != 0)
			init_W();

		if (no_bias) {
			Parameter b({}, "b");
			register_params("b", b);
		} else {
			Tensor b_data(out_channels);
			Parameter b(b_data, "b");
			register_params("b", b);
		}
	}

	void Conv2d::init_W() {
		size_t C = this->in_channels;
		size_t OC = this->out_channels;
		auto [KH, KW] = kernel_size;

		float scale = std::sqrt(1 / static_cast<float>(C * KH * KW));
		Tensor W_data = randn({OC, C, KH, KW}) * scale;
		params["W"].data() = W_data;
	}

	Variable Conv2d::forward(const std::vector<Variable>& xs) {
		const Variable& x = xs[0];
		const Parameter& W = get_param("W");
		const Parameter& b = get_param("b");
		if (W.data().empty()) {
			in_channels = x.shape()[1];
			init_W();
		}
/*		for (size_t i =0; i < x.shape().size(); i++) 
			std::cout << x.shape()[i] << " ";
		std::cout << std::endl;
		for (size_t i =0; i < W.shape().size(); i++) 
			std::cout << W.shape()[i] << " ";
		std::cout << std::endl;
		for (size_t i =0; i < b.shape().size(); i++) 
			std::cout << b.shape()[i] << " ";
		std::cout << std::endl;
*/		
		Variable y = conv2d(x, W, b, stride, pad);
		return y;
	}
}
