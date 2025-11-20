#include "dataset/dataset.hpp"
#include "dataset/utils.hpp"
#include "container/tensor/tensor_all.hpp"

#include <cmath>
#include <random>
#include <string>
#include <map>

#ifdef _WIN32
#include <winsock2.h>
#else 
#include <arpa/inet.h>
#endif

using namespace tensor;

// [Dataset]
Tensor<> Dataset::get_data() const {
	return data;
}

Tensor<> Dataset::get_label() const {
	return label;

}

Tensor<> Dataset::get_data(size_t index) const {
	Tensor<> row = data[index];
	if (transform)
		return (*transform)(row);
	return row;
}

Tensor<> Dataset::get_label(size_t index) const {
	Tensor<> row = label[index];
	if (target_transform)
		return (*target_transform)(row);
	return row;
}

// [SpiralDataset]
SpiralDataset::SpiralDataset(size_t num_data,
							size_t num_class,
							bool train)
		: Dataset(train),
		num_data(num_data),
		num_class(num_class) {
	init_dataset();
}

void SpiralDataset::init_dataset() {
	const size_t input_dim = 2;
	const size_t total = num_data * num_class;
	const size_t seed = train ? 1984 : 2020;

	Tensor<> x_data({total, input_dim});
	Tensor<> t_data({total});
	Tensor<> noise = randn(total, 1, seed);

	for (size_t j = 0; j < num_class; j++) {
		for (size_t i = 0; i < num_data; i++) {
			float rate = static_cast<float>(i) / num_data;
			float r = 1.0f * rate;
			size_t ix = j * num_data + i;

			float theta = j * 4.0f + 4.0f * rate + noise({ix, 0}) * 0.2f;

			x_data({ix, 0}) = r * std::sin(theta);
			x_data({ix, 1}) = r * std::cos(theta);
			t_data({ix}) = static_cast<float>(j);
		}
	}

	Tensor<size_t> indices = permutation(total);

	Tensor<> x_shuffled({total, input_dim});
	Tensor<> t_shuffled({total});

	for (size_t i = 0; i < total; i++) {
		for (size_t d = 0; d < input_dim; d++)
			x_shuffled({i, d}) = x_data({indices({i}), d});
		t_shuffled({i}) = t_data({indices({i})});
	}

	data = x_shuffled;
	label = t_shuffled;
}

// [BigDataset]
BigDataset::BigDataset(	size_t num_data,
						size_t num_class,
						bool train)
		: Dataset(train),
		num_data(num_data),
		num_class(num_class) {}

Tensor<> BigDataset::get_data(size_t index) {
	//TODO: load data
	std::string csv_file = "big_data_" + std::to_string(index) + ".csv";
	data = Tensor<>::from_csv(csv_file, false, false);	
	return data;
}

Tensor<> BigDataset::get_label(size_t index) {
	std::string csv_file = "big_label_" + std::to_string(index) + ".csv";
	label = Tensor<>::from_csv(csv_file, false, false);	
	return label;
}


// [MNISTDataset]
MNISTDataset::MNISTDataset(bool train) : Dataset(train) {	
	transform = std::make_shared<Compose<float>>(
		std::vector<std::shared_ptr<Transform<float>>>({
			std::make_shared<Flatten<float>>(), 
			std::make_shared<ToFloat<float>>(), 
			std::make_shared<Normalize<float>>(0, 255)
		}));

	init_dataset();

}

Tensor<> MNISTDataset::_load_data(std::string& file_path) {
	std::ifstream file(file_path, std::ios::binary);
	if(!file) throw std::runtime_error("Cannot open file: " + file_path);

	uint32_t magic, num, rows, cols;
	file.read((char*)&magic, 4);
	file.read((char*)&num, 4);
	file.read((char*)&rows, 4);
	file.read((char*)&cols, 4);

	magic = ntohl(magic);
	num = ntohl(num);
	rows = ntohl(rows);
	cols = ntohl(cols);

	std::vector<float> buffer(num * rows * cols);
	for (size_t i = 0; i < buffer.size(); i++) {
		unsigned char byte;
		file.read((char*)&byte, 1);
		buffer[i] =static_cast<float>(byte);
	}

	return Tensor<>({num, rows, cols}, buffer);
}


Tensor<> MNISTDataset::_load_label(std::string& file_path) {
	std::ifstream file(file_path, std::ios::binary);
	if(!file) throw std::runtime_error("Cannot open file: " + file_path);

	uint32_t magic, num;
	file.read((char*)&magic, 4);
	file.read((char*)&num, 4);

	magic = ntohl(magic);
	num = ntohl(num);

	std::vector<uint8_t> raw_labels(num);
	file.read((char*)raw_labels.data(), num);

	std::vector<float> labels(num);
	for (size_t i = 0; i < num; i++)
		labels[i] = static_cast<float>(raw_labels[i]);

	return Tensor<>({num}, labels);
}

void MNISTDataset::init_dataset() {
	std::string base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"; 

	std::map<std::string, std::string> train_files = {
		{"target", "train-images-idx3-ubyte.gz"},
		{"label", "train-labels-idx1-ubyte.gz"}
	};
	std::map<std::string, std::string> test_files = {
		{"target", "t10k-images-idx3-ubyte.gz"},
		{"label", "t10k-labels-idx1-ubyte.gz"}
	};

	const auto& files = train ? train_files : test_files;

	// get_dataset_file, gunzip_file - dataset/utils.hpp
	std::string data_gz_path = get_dataset_file(base_url + files.at("target"));
	std::string label_gz_path = get_dataset_file(base_url + files.at("label"));

	std::string data_path = gunzip_file(data_gz_path);
	std::string label_path = gunzip_file(label_gz_path);

	data = _load_data(data_path);
	label = _load_label(label_path);
}

void MNISTDataset::show(size_t index) {
    if (index >= data.size())
        throw std::out_of_range("Index out of range in MNISTDataset::show");

    const Tensor<float>& image = data[index];  // shape: [28, 28] 혹은 [1, 28, 28]
    std::vector<size_t> shape = image.get_shape();

    // 이미지가 [1, 28, 28]인 경우 첫 축 제거
    size_t H = shape.back() == 28 ? shape[shape.size() - 2] : shape[0];
    size_t W = 28;

    std::cout << "Label: " << label[index]({0}) << std::endl;

    for (size_t i = 0; i < H; ++i) {
        for (size_t j = 0; j < W; ++j) {
            float val = (shape.size() == 3)
                ? image({0, i, j})
                : image({i, j});
            char pixel = val > 204 ? '#' :
                         val > 127 ? '+' :
                         val > 51 ? '.' : ' ';
            std::cout << pixel;
        }
        std::cout << "\n";
    }

}
