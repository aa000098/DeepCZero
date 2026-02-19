#pragma once

#include "container/parameter.hpp"
#include "container/tensor/tensor_all.hpp"
#include "utils/utils.hpp"

#include <unordered_map>
#include <map>
#include <string>
#include <memory>
#include <vector>
#include <cassert>

class Parameter;

namespace cnpy {
	struct NpyArray;
	using npz_t = std::map<std::string, NpyArray>;
}

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
		void set_param_data(const std::string& name, const Tensor<>& data);
		std::shared_ptr<Layer> get_sublayer(const std::string& name) const;

		Variable operator()(const std::vector<Variable>& inputs);
		Variable operator()(const std::initializer_list<Variable>& inputs);
		Variable operator()(const Variable& input);

		void cleargrads();

		std::vector<Parameter> get_params();

		std::unordered_map<std::string, Parameter> flatten_params(const std::string& parent_key = "");
		
		void save_weights(const std::string& path);

		void load_weights(const std::string& path);

		// Load W and b from npz file for this layer
		void load_params_from_npz(const std::string& npz_path, const std::string& layer_name);
		void load_params_from_npz(const cnpy::npz_t& npz, const std::string& layer_name);

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
		std::pair<size_t, size_t> kernel_size;
		std::pair<size_t, size_t> stride;
		std::pair<size_t, size_t> pad;
		bool no_bias;
		size_t in_channels;
		// dtype

	public:
		Conv2d() = default;
		Conv2d(size_t out_channels,
				std::pair<size_t, size_t> kernel_size,
				std::pair<size_t, size_t> stride = {1, 1},
				std::pair<size_t, size_t> pad = {0, 0},
				bool no_bias = false,
				size_t in_channels = 0);

		Conv2d(size_t out_channels,
				std::initializer_list<size_t> kernel_size,
				std::initializer_list<size_t> stride = {1,1},
				std::initializer_list<size_t> pad = {0,0},
				bool no_bias = false,
				size_t in_channels = 0) 
			: Conv2d(out_channels,
					to_pair(kernel_size),
					to_pair(stride),
					to_pair(pad),
					no_bias,
					in_channels) {}

		void init_W();
		Variable forward(const std::vector<Variable>& xs) override;

	};

	class BatchNorm2d : public Layer {
	private:
		size_t num_features;
		float momentum;
		float eps;
		Tensor<> running_mean;
		Tensor<> running_var;

	public:
		BatchNorm2d() = default;
		BatchNorm2d(size_t num_features,
					float momentum = 0.1f,
					float eps = 1e-5f);

		Variable forward(const std::vector<Variable>& xs) override;

		const Tensor<>& get_running_mean() const { return running_mean; }
		const Tensor<>& get_running_var() const { return running_var; }
		void set_running_mean(const Tensor<>& t) { running_mean = t; }
		void set_running_var(const Tensor<>& t) { running_var = t; }
	};

	class RNN : public Layer {
	private:
		std::shared_ptr<Layer> x2h;
		std::shared_ptr<Layer> h2h;
		Variable hidden_state;

	public:
		RNN(size_t hidden_size, size_t input_size = 0);
		void reset_state() {
			hidden_state = Variable();
		}
		Variable forward(const std::vector<Variable>& xs) override;

	};
}


