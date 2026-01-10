#include "container/layer/layer.hpp"
#include "function/function.hpp"
#include "function/activation_functions.hpp"
#include "utils/io.hpp"
#include "cnpy.h"

using function::Sigmoid;

namespace layer {
	Parameter Layer::get_param(const std::string& name) const {
		auto it = params.find(name);
		if (it != params.end()) return it->second;
		else throw std::runtime_error("Parameter not found: " + name);
	}

	void Layer::set_param_data(const std::string& name, const Tensor<>& data) {
		auto it = params.find(name);
		if (it != params.end()) {
			it->second.data() = data;
		} else {
			throw std::runtime_error("Parameter not found: " + name);
		}
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
		this->inputs = {input};
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

	void Layer::save_weights(const std::string& path) {
		// 1) 모든 파라미터 평탄화
		std::unordered_map<std::string, Parameter> params_dict = flatten_params();

		// weights 디렉토리 생성
		std::filesystem::path weights_dir = get_cache_file_path("weights");
		std::filesystem::create_directories(weights_dir);

		std::string out_path = (weights_dir / path).string();
		std::ofstream fout(out_path, std::ios::binary);
		if (!fout.is_open()) {
			throw std::runtime_error("Cannot open file for writing: " + out_path);
		}

		// 2) magic + 파라미터 개수 기록
		const char magic[4] = {'D', 'C', 'Z', '1'};
		fout.write(magic, 4);

		uint32_t num_params = static_cast<uint32_t>(params_dict.size());
		fout.write(reinterpret_cast<char*>(&num_params), sizeof(num_params));

		// 3) 각 파라미터 쓰기
		for (auto& kv : params_dict) {
			const std::string& name = kv.first;
			Parameter& param = kv.second;
			Tensor<>& t = param.data();

			// 비어있는 텐서는 건너뛰고 싶으면 여기서 continue 해도 됨
			// if (t.empty()) continue;

			// 이름
			uint32_t name_len = static_cast<uint32_t>(name.size());
			fout.write(reinterpret_cast<char*>(&name_len), sizeof(name_len));
			fout.write(name.data(), name_len);

			// shape
			std::vector<size_t> shape = t.get_shape();
			uint32_t ndim = static_cast<uint32_t>(shape.size());
			fout.write(reinterpret_cast<char*>(&ndim), sizeof(ndim));

			uint64_t numel = 1;
			for (size_t d : shape) numel *= static_cast<uint64_t>(d);

			for (size_t d : shape) {
				uint64_t dim = static_cast<uint64_t>(d);
				fout.write(reinterpret_cast<char*>(&dim), sizeof(dim));
			}

			// data (float32)
			// Tensor가 view일 수도 있으니, contiguous로 한 번 복사하는 게 안전
			Tensor<> contig = t.contiguous();
			const std::vector<float>& data = contig.raw_data();

			if (data.size() != numel) {
				throw std::runtime_error("Tensor size mismatch when saving param " + name);
			}

			fout.write(reinterpret_cast<const char*>(data.data()),
					   sizeof(float) * data.size());
		}

		if (!fout.good()) {
			throw std::runtime_error("Error occurred while writing weights file: " + path);
		}
	}

	void Layer::load_weights(const std::string& path) {
		std::ifstream fin(path, std::ios::binary);
		if (!fin.is_open()) {
			throw std::runtime_error("Cannot open file for reading: " + path);
		}

		// 1) magic 검사
		char magic[4];
		fin.read(magic, 4);
		if (!fin.good() || magic[0] != 'D' || magic[1] != 'C' ||
			magic[2] != 'Z' || magic[3] != '1') {
			throw std::runtime_error("Invalid weight file (magic mismatch): " + path);
		}

		// 2) 파라미터 개수
		uint32_t num_params = 0;
		fin.read(reinterpret_cast<char*>(&num_params), sizeof(num_params));

		// 3) 파일에서 읽은 텐서들을 임시 맵에 저장
		std::unordered_map<std::string, Tensor<>> loaded_tensors;

		for (uint32_t i = 0; i < num_params; ++i) {
			// 이름
			uint32_t name_len = 0;
			fin.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
			std::string name(name_len, '\0');
			fin.read(&name[0], name_len);

			// shape
			uint32_t ndim = 0;
			fin.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));

			std::vector<size_t> shape(ndim);
			uint64_t numel = 1;
			for (uint32_t k = 0; k < ndim; ++k) {
				uint64_t dim = 0;
				fin.read(reinterpret_cast<char*>(&dim), sizeof(dim));
				shape[k] = static_cast<size_t>(dim);
				numel *= dim;
			}

			// data
			std::vector<float> data(numel);
			fin.read(reinterpret_cast<char*>(data.data()),
					 sizeof(float) * numel);

			if (!fin.good()) {
				throw std::runtime_error("Error reading data for param: " + name);
			}

			Tensor<> t(shape, data);
			loaded_tensors.emplace(name, std::move(t));
		}

		// 4) 현재 Layer의 params에 매핑
		std::unordered_map<std::string, Parameter> params_dict = flatten_params();

		for (auto& kv : params_dict) {
			const std::string& name = kv.first;
			Parameter& param = kv.second;

			auto it = loaded_tensors.find(name);
			if (it == loaded_tensors.end()) {
				// 저장 당시에는 있었는데 지금 구조에서 빠졌다면: 경고 정도만
				// std::cerr << "Warning: no weight found for param " << name << std::endl;
				continue;
			}

			// shape이 다르면 여기서 체크해주는 것도 좋다.
			const auto& loaded = it->second;
			if (param.data().empty()) {
				// 아직 init 안 된 상태면 그냥 대입
				param.data() = loaded;
			} else {
				// 이미 shape가 있다면 일치 여부 확인
				if (param.data().get_shape() != loaded.get_shape()) {
					throw std::runtime_error("Shape mismatch for param " + name);
				}
				param.data() = loaded;
			}
		}
	}

	void Layer::load_params_from_npz(const std::string& npz_path, const std::string& layer_name) {
		cnpy::npz_t npz = cnpy::npz_load(npz_path);
		load_params_from_npz(npz, layer_name);
	}

	void Layer::load_params_from_npz(const cnpy::npz_t& npz, const std::string& layer_name) {
		// Load weight (W)
		std::string w_key = layer_name + ".W";
		auto w_it = npz.find(w_key);
		if (w_it != npz.end()) {
			const cnpy::NpyArray& w_arr = w_it->second;
			std::vector<float> w_data = w_arr.as_vec<float>();
			Tensor<> w_tensor(w_arr.shape, w_data);
			set_param_data("W", w_tensor);
		}

		// Load bias (b)
		std::string b_key = layer_name + ".b";
		auto b_it = npz.find(b_key);
		if (b_it != npz.end()) {
			const cnpy::NpyArray& b_arr = b_it->second;
			std::vector<float> b_data = b_arr.as_vec<float>();
			Tensor<> b_tensor(b_arr.shape, b_data);
			set_param_data("b", b_tensor);
		}
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
