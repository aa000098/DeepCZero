#pragma once

#include "container/tensor/tensor.hpp"

namespace tensor {

template<typename T>
Tensor<T> add(const Tensor<T>& a, const Tensor<T>& b) {
	if (a.get_shape() != b.get_shape())
		throw std::runtime_error("Shape mismatch in tensor add");

	std::vector<T> result_data(a.raw_data());
	const std::vector<T>& b_data = b.raw_data();
	for (size_t i = 0; i < result_data.size(); i++)
		result_data[i] += b_data[i];

	return Tensor<T>(a.get_shape(), result_data);
}

template<typename T>
Tensor<T> sub(const Tensor<T>& a, const Tensor<T>& b) {
	if (a.get_shape() != b.get_shape())
		throw std::runtime_error("Shape mismatch in tensor subtract");

	std::vector<T> result_data(a.raw_data());
	const std::vector<T>& b_data = b.raw_data();
	for (size_t i = 0; i < result_data.size(); ++i)
		result_data[i] -= b_data[i];

	return Tensor<T>(a.get_shape(), result_data);
}

template<typename T>
Tensor<T> mul(const Tensor<T>& a, const Tensor<T>& b) {
	if (a.get_shape() != b.get_shape())
		throw std::runtime_error("Shape mismatch in tensor subtract");

	std::vector<T> result_data(a.raw_data());
	const std::vector<T>& b_data = b.raw_data();
	for (size_t i = 0; i < result_data.size(); ++i)
		result_data[i] *= b_data[i];

	return Tensor<T>(a.get_shape(), result_data);
}

template<typename T>
Tensor<T> neg(const Tensor<T>& a) {
	std::vector<T> result_data(a.raw_data());
	for (auto& val : result_data)
		val = -val;

	return Tensor<T>(a.get_shape(), result_data);
}

// inplace
template<typename T>
void add_inplace(Tensor<T>& a, const Tensor<T>& b) {
	auto& data = a.raw_data();
	const auto& b_data = b.raw_data();
	if (a.get_shape() != b.get_shape())
		throw std::runtime_error("Shape mismatch in add_inplace");
	for (size_t i = 0; i < data.size(); ++i)
		data[i] += b_data[i];
}

template<typename T>
void sub_inplace(Tensor<T>& a, const Tensor<T>& b) {
	auto& data = a.raw_data();
	const auto& b_data = b.raw_data();
	if (a.get_shape() != b.get_shape())
		throw std::runtime_error("Shape mismatch in sub_inplace");
	for (size_t i = 0; i < data.size(); ++i)
		data[i] -= b_data[i];
}

template<typename T>
void mul_inplace(Tensor<T>& a, const Tensor<T>& b) {
	auto& data = a.raw_data();
	const auto& b_data = b.raw_data();
	if (a.get_shape() != b.get_shape())
		throw std::runtime_error("Shape mismatch in mul_inplace");
	for (size_t i = 0; i < data.size(); ++i)
		data[i] *= b_data[i];
}

// Tensor ⊕ Scalar
template<typename T>
void add_scalar_inplace(Tensor<T>& a, T scalar) {
	auto& data = a.raw_data();
	for (auto& val : data)
		val += scalar;
}

// Tensor<T> -= scalar
template<typename T>
void sub_scalar_inplace(Tensor<T>& a, T scalar) {
	auto& data = a.raw_data();
	for (auto& val : data)
		val -= scalar;
}

template<typename T>
void mul_scalar_inplace(Tensor<T>& a, T scalar) {
	auto& data = a.raw_data();
	for (auto& val : data)
		val *= scalar;
}

template<typename T>
void div_scalar_inplace(Tensor<T>& a, T scalar) {
	auto& data = a.raw_data();
	for (auto& val : data)
		val /= scalar;
}


}
