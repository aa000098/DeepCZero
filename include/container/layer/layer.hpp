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

		void register_params(
				const std::string& name, 
				const Parameter& param) {
			params[name] = param;
		}

		virtual Variable forward(const std::vector<Variable>& xs) = 0;
	
		Parameter get_param(const std::string& name) const {
			auto it = params.find(name);
			if (it != params.end()) return it->second;
			else throw std::runtime_error("Parameter not found: " + name);
		}

		Variable operator()(const std::vector<Variable>& inputs) {
			Variable output = forward(inputs);
			this->inputs = inputs;
			this->output = output;
			return output;
		}

		void cleargrad() {
			for (auto& pair : params)
				pair.second.cleargrad();
		}

		std::unordered_map<std::string, Parameter> get_params() { return params; }
		

	};

	class Linear : public Layer {
	private:
		size_t in_size;
		size_t out_size;

	public:
		Linear( size_t out_size, 
				bool nobias = false,
				/*dtype = float32, */
				size_t in_size = 0) 
			: in_size(in_size), out_size(out_size) {
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

		Linear() = default;

		void init_W() {
			Tensor W_data = randn(in_size, out_size); 
			W_data *= std::sqrt(1/in_size);
			params["W"].data() = W_data;
		}

		Variable forward(const std::vector<Variable>& xs) {
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

	};

}
