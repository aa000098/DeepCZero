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
	
		Parameter get_param(const std::string& name) const;

		Variable operator()(const std::vector<Variable>& inputs);
		Variable operator()(const std::initializer_list<Variable>& inputs);
		Variable operator()(const Variable& input);

		void cleargrad();

		std::unordered_map<std::string, Parameter>& get_params() { return params; }
		

	};

	class Linear : public Layer {
	private:
		size_t in_size;
		size_t out_size;

	public:
		Linear( size_t out_size, 
				bool nobias = false,
				/*dtype = float32, */
				size_t in_size = 0);

		Linear() = default;

		void init_W();

		Variable forward(const std::vector<Variable>& xs);

	};

}
