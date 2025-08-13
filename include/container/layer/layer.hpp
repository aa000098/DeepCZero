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

		std::unordered_map<std::string, Parameter> flatten_params(const std::string& parent_key = "");
		
		void save_weights(std::string path);

		void load_weights(std::string path);

	};

	class Linear : public Layer {
	private:
		size_t out_size;
		size_t in_size;

	public:
		Linear( size_t out_size, 
				bool nobias = false,
				/*dtype = float32, */
				size_t in_size = 0);

		Linear() = default;

		void init_W();

		Variable forward(const std::vector<Variable>& xs) override;

	};

	class Conv2d : public Layer {
	private:
		size_t out_channels;
		size_t in_channels;
		std::pair<size_t, size_t> kernel_size;
		std::pair<size_t, size_t> stride;
		std::pair<size_t, size_t> pad;
		bool no_bias;
		// dtype

	public:
		Conv2d() = default;
		Conv2d(size_t out_channels,
				std::pair<size_t, size_t> kernel_size,
				std::pair<size_t, size_t> stride = {1, 1},
				std::pair<size_t, size_t> pad = {0, 0},
				bool no_biasi = false,
				size_t in_channels = 0);

		void init_W();
		Variable forward(const std::vector<Variable>& xs) override;

	};
}	
