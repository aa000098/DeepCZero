#pragma once

#include "container/parameter.hpp"
#include "container/tensor/tensor_all.hpp"

#include <unordered_map>
#include <string>
#include <memory>
#include <vector>

class Parameter;

namespace layer {

	class Layer {
	protected:
		std::unordered_map<std::string, Parameter> params;
		std::vector<Variable> inputs;
		Variable output;

	public:

		void register_params(const std::string& name, Parameter& param) {
			params[name] = param;
		}

		virtual Variable forward(const std::vector<Variable>& xs) = 0;
	
		Parameter get_param(const std::string& name) const {
			auto it = params.find(name);
			if (it != params.end()) return it->second;
			else throw std::runtime_error("Parameter not found: " + name);
		}

		Variable operator()(std::vector<Variable>& inputs) {
			Variable output = forward(inputs);
			this->inputs = inputs;
			this->output = output;
			return output;
		}

		void cleargrads() {
			for (auto& pair : params)
				pair.second.cleargrad();
		}

	};

	class Linear : public Layer {
	private:
		Parameter W;
		Parameter b;
		size_t in_size;
		size_t out_size;

	public:
		Linear(size_t in_size, 
					size_t out_size, 
					bool nobias = false
					/*, dtype = float32 */) 
			: in_size(in_size), out_size(out_size) {
			if (nobias)
				b = Parameter();
		};

		Linear() = default;

		Variable forward(const std::vector<Variable>& xs) {
			if (!W.data().empty()) {
				in_size = xs[0].shape()[1];
			}
			Variable y = linear(xs[0], W, b);
			return y;
		}

	};

}
