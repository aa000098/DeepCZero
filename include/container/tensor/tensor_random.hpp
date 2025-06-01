#include "container/tensor/tensor_all.hpp"

#include <random>

inline Tensor<> rand(size_t rows, size_t cols, size_t seed) {
	std::mt19937 gen(seed);

	std::uniform_real_distribution<> dist(0, 1);

	std::vector<float> data;
	data.reserve(rows * cols);
	for (size_t i = 0; i < rows * cols; i++)
		data.push_back(dist(gen));

	return Tensor<>({rows, cols}, data);
}

inline Tensor<> rand(size_t rows, size_t cols) {
	std::random_device rd;
	return rand(rows, cols, rd());
}


inline Tensor<> randn(size_t rows, size_t cols, size_t seed) {
	std::mt19937 gen(seed);

	std::normal_distribution<> dist(0, 1);

	std::vector<float> data;
	data.reserve(rows * cols);
	for (size_t i = 0; i < rows * cols; i++)
		data.push_back(dist(gen));

	return Tensor<>({rows, cols}, data);
}

inline Tensor<> randn(size_t rows, size_t cols) {
	std::random_device rd;
	return randn(rows, cols, rd());
}
