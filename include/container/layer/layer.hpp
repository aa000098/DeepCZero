#pragma once

#include "container/parameter.hpp"
#include "container/tensor/tensor_all.hpp"

#include "function/function_all.hpp"

#include <unordered_map>
#include <string>
#include <memory>
#include <vector>

using function::Sigmoid;

class Parameter;

namespace layer {

	class Layer {
	protected:
		std::unordered_map<std::string, Parameter> params;
		std::unordered_map<std::string, std::shared_ptr<Layer>> sublayers;
		std::vector<Variable> inputs;
		Variable output;

	public:

		void register_params(
				const std::string& name, 
				const Parameter& param) {
			params[name] = param;
		}

		void register_sublayers(
				const std::string& name,
				const std::shared_ptr<Layer>& layer) {
			sublayers[name] = layer;
		}

		virtual Variable forward(const std::vector<Variable>& xs) = 0;
	
		Parameter get_param(const std::string& name) const;
		std::shared_ptr<Layer> get_sublayer(const std::string& name) const;

		Variable operator()(const std::vector<Variable>& inputs);
		Variable operator()(const std::initializer_list<Variable>& inputs);
		Variable operator()(const Variable& input);

		void cleargrads();

		std::vector<Parameter> get_params();
		

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

		Variable forward(const std::vector<Variable>& xs) override;

	};
	
	class MLP : public Layer {
	private:
		std::shared_ptr<Function> activation;
		std::vector<std::shared_ptr<Layer>> layers;

	public:
		MLP(const std::vector<size_t>& fc_output_sizes, 
			const std::shared_ptr<Function> activation
			= std::make_shared<Sigmoid>()) : activation(activation) {
		size_t i = 0;
		for (size_t out_size : fc_output_sizes) {
			std::shared_ptr<Layer> layer = std::make_shared<Linear>(out_size);
			register_sublayers("l" + std::to_string(i), layer);
			Linear l(out_size);
			this->layers.push_back(layer);
			i++;
		}
	}

		MLP(const std::initializer_list<size_t>& fc_output_sizes,
			const std::shared_ptr<Function> activation
			 = std::make_shared<Sigmoid>()) : MLP(std::vector<size_t>(fc_output_sizes), activation) {};

		Variable forward(const std::vector<Variable>& xs) override;
	};

}
